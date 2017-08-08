
#ifndef MXNET_OPERATOR_REORDER_INL_H_ 
#define MXNET_OPERATOR_REORDER_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h> 
#include <mxnet/operator.h> 
#include <cstring>
#include <algorithm>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "./operator_common.h"
#include "./mshadow_op.h"

namespace mxnet {
namespace op {

// Declare enumeration of input order to make code more intuitive.
// These enums are only visible within this header
namespace reord {
enum WeightedFusionOpInputs {kData};
enum WeightedFusionOpOutputs {kOut};
enum WeightedFusionOpResource { kTempSpace };
}  // reord

struct ReorderParam : public dmlc::Parameter<ReorderParam> {
  uint32_t branch_factor;
  uint64_t workspace;
  DMLC_DECLARE_PARAMETER(ReorderParam) {
	  DMLC_DECLARE_FIELD(branch_factor).set_lower_bound(1)
		  .describe("Number of branches to be summed.");
	  DMLC_DECLARE_FIELD(workspace).set_default(1024).set_range(0, 8192)
		  .describe("Tmp workspace for convolution (MB).");
  }
};


template<typename xpu, typename DType>
class ReorderOp : public Operator {
 public:
	 explicit ReorderOp(ReorderParam param)
		 : size_(param.branch_factor), workspace_((param.workspace << 20) / sizeof(DType))  {
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(static_cast<int>(in_data.size()), 1);
    CHECK_EQ(out_data.size(), 1);
    if (req[reord::kOut] == kNullOp) return;
    Stream<xpu> *s = ctx.get_stream<xpu>();
	Tensor<xpu, 4, DType> data = in_data[reord::kData].get<xpu, 4, DType>(s);
	Shape<4> data_shape = data.shape_;
	Tensor<xpu, 4, DType> out = out_data[reord::kOut].get<xpu, 4, DType>(s);

	Tensor<xpu, 1, DType> workspace =
		ctx.requested[reord::kTempSpace].get_space_typed<xpu, 1, DType>(
		Shape1(this->InitTemp(data.shape_, out.shape_)), s);
	const index_t nbatch = data.size(0);
	for (index_t i = 0; i < nbatch; i += nstep_) {
		const index_t step = std::min(nstep_, nbatch - i);
		Tensor<xpu, 3, DType> temp_data = Tensor<xpu, 3, DType>(workspace.dptr_,
			mshadow::Shape3(shape_dstunit_[0], shape_dstunit_[1],
			shape_dstunit_[2] * step), s);
		Tensor<xpu, 4, DType> temp_out = Tensor<xpu, 4, DType>(
			workspace.dptr_ + temp_data.shape_.Size(),
			mshadow::Shape4(out.shape_[1], step, out.shape_[2], out.shape_[3]), s);
		temp_data = reshape(swapaxis<1, 0>(data.Slice(i,i+step)), temp_data.shape_);
		temp_out = reshape(swapaxis<1, 0>(temp_data), mshadow::Shape4(out.shape_[1], step, out.shape_[2], out.shape_[3]));
		out.Slice(i, i + step) = swapaxis<1, 0>(temp_out);
	}
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_grad.size(), static_cast<size_t>(1));
    Stream<xpu> *s = ctx.get_stream<xpu>();
	Tensor<xpu, 4, DType> grad_data = in_grad[reord::kData].get<xpu, 4, DType>(s);
	Tensor<xpu, 4, DType> grad_out = out_grad[reord::kOut].get<xpu, 4, DType>(s);


	Tensor<xpu, 1, DType> workspace =
		ctx.requested[reord::kTempSpace].get_space_typed<xpu, 1, DType>(
		Shape1(this->InitTemp(grad_data.shape_, grad_out.shape_)), s);
	const index_t nbatch = grad_data.size(0);
	for (index_t i = 0; i < nbatch; i += nstep_) {
		const index_t step = std::min(nstep_, nbatch - i);
		Tensor<xpu, 3, DType> temp_data = Tensor<xpu, 3, DType>(workspace.dptr_,
			mshadow::Shape3(shape_dstunit_[1], shape_dstunit_[0],
			shape_dstunit_[2] * step), s);
		Tensor<xpu, 4, DType> temp_out = Tensor<xpu, 4, DType>(
			workspace.dptr_ + temp_data.shape_.Size(),
			mshadow::Shape4(grad_out.shape_[1], step, grad_out.shape_[2], grad_out.shape_[3]), s);
		temp_data = reshape(swapaxis<1, 0>(grad_out.Slice(i, i + step)), temp_data.shape_);
		temp_out = reshape(swapaxis<1, 0>(temp_data), mshadow::Shape4(grad_out.shape_[1], step, grad_out.shape_[2], grad_out.shape_[3]));
		grad_data.Slice(i, i + step) = swapaxis<1, 0>(temp_out);
	}
  }
  inline void Save(dmlc::JSONWriter *writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("size_", size_);
    writer->EndObject();
  }
  inline void Load(dmlc::JSONReader *reader) {
    dmlc::JSONObjectReadHelper helper;
    helper.DeclareField("size_", &size_);
    helper.ReadAllFields(reader);
  }

 private:
	 inline index_t InitTemp(const mshadow::Shape<4> &ishape,
		 const mshadow::Shape<4> &oshape) {
		 shape_dstunit_ = mshadow::Shape3(size_,
			 ishape[1] / size_, ishape[2] * ishape[3]);
		 // param_.workspace is in elements of sizeof(DType)
		 // if param_.workspace is set to zero the nstep_ equals ishape[0] (batch)
		 nstep_ = std::max(
			 std::min(
			 static_cast<index_t>(
			 workspace_ / shape_dstunit_.Size()),
			 ishape[0]),
			 1U);
		 mshadow::Shape<3> sdst = mshadow::Shape3(shape_dstunit_[0], shape_dstunit_[1],
			 shape_dstunit_[2] * nstep_);
		 index_t required_size = 2 * sdst.Size();
		 CHECK_GE(workspace_, required_size)
			 << "\nMinimum workspace size: " << required_size * sizeof(DType) << " Bytes\n"
			 << "Given: " << workspace_ * sizeof(DType) << " Bytes";
		 return required_size;
	 }
	 uint32_t size_;
	 uint64_t workspace_;
	 mshadow::Shape<3> shape_dstunit_;
	 index_t nstep_;
};  // class ReorderOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(ReorderParam param, int dtype);

#if DMLC_USE_CXX11
class ReorderProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
	  return { "data"};
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    //checkout input datas are in same shape
	CHECK_EQ(in_shape->size(), static_cast<size_t>(1));
	TShape dshape = in_shape->at(reord::kData);
	if (dshape.ndim() == 0) return false;
	CHECK_EQ(dshape[1] % param_.branch_factor, 0);
    //setup the output shape
    out_shape->clear();
    out_shape->push_back(in_shape->at(0));
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    for (index_t i = 0; i < in_type->size(); ++i) {
      if ((*in_type)[i] == -1) {
        (*in_type)[i] = dtype;
      } else {
        CHECK_EQ((*in_type)[i], dtype) << "This layer requires uniform type. "
                                       << "Expected " << dtype << " v.s. given "
                                       << (*in_type)[i] << " at " << ListArguments()[i];
      }
    }
    out_type->clear();
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
	  ReorderProp* ptr = new ReorderProp();
	  ptr->param_ = this->param_;
	  return ptr;
  }

  std::string TypeString() const override {
    return "Reorder";
  }

  // decalre dependency and inplace optimization options
  //std::vector<int> DeclareBackwardDependency(
  //  const std::vector<int> &out_grad,
  //  const std::vector<int> &in_data,
  //  const std::vector<int> &out_data) const override {
  //  return {out_grad[wfuse::kOut], in_data[wfuse::kData]};
  //}

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
	  const std::vector<int> &in_data,
	  const std::vector<void*> &out_data) const override {
	  return{ { in_data[reord::kData], out_data[reord::kOut] } };
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
	  return{ { out_grad[reord::kOut], in_grad[reord::kData] } };
  }

  std::vector<int> DeclareBackwardDependency(
	  const std::vector<int> &out_grad,
	  const std::vector<int> &in_data,
	  const std::vector<int> &out_data) const override {
	  return{ out_grad[reord::kOut]};
  }

  std::vector<ResourceRequest> ForwardResource(
	  const std::vector<TShape> &in_shape) const override {
	  return{ ResourceRequest::kTempSpace };
  }

  std::vector<ResourceRequest> BackwardResource(
	  const std::vector<TShape> &in_shape) const override {
	  return{ ResourceRequest::kTempSpace };
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
	 ReorderParam param_;
};  // class ReorderSymbol
#endif
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_REORDER_INL_H_
