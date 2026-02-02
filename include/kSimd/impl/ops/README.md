# ops
这个文件夹是对基础指令的封装，不涉及高级数学运算。
高级数学运算放到 extension 模块

# 设计原则
## 类继承
每一个 SimdOp<Instruction, ScalarType> 只能编写 **<= Instruction** 的指令(SSE op 中不能含有SSE2指令)。
SSE2 op 要继承 SSE op，来复用比他更加低级的指令，以及使用更高级的指令覆盖掉父类的低级指令。

## 函数
op类不能出现任何的虚函数，全部函数必须 force inline
