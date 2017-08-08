
#include "./reorder-inl.h"
namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(ReorderParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
	  op = new ReorderOp<gpu, DType>(param);
  })
  return op;
}
}  // namespace op
}  // namespace mxnet