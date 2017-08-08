#include "./reorder-inl.h"
namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(ReorderParam param, int dtype) {
  Operator *op = NULL;
  switch (dtype) {
  case mshadow::kFloat32:
	  op = new ReorderOp<cpu, float>(param);
    break;
  case mshadow::kFloat64:
	  op = new ReorderOp<cpu, double>(param);
    break;
  case mshadow::kFloat16:
    LOG(FATAL) << "float16 fully connected layer is currently"
                  "only supported by CuDNN version.";
    break;
  default:
    LOG(FATAL) << "Unsupported type " << dtype;
  }
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *ReorderProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                     std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(ReorderParam);

MXNET_REGISTER_OP_PROPERTY(Reorder, ReorderProp)
.add_argument("data", "Symbol", "Input data to the WeightedFusionOp.")
.describe("Perform an weighted sum over all the inputs.")
.add_arguments(ReorderParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet