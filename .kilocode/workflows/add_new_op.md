# add_new_op.md

这个工作流仅用于添加新的op(添加新的Intrinsics封装函数)

## 步骤

1. 阅读 @kSimd/kSimd/core/impl/base.hpp 文件，深刻理解 KSIMD_DYN_INSTRUCTION_* 和 KSIMD_DYN_DISPATCH_LEVEL_* 等宏的含义
2. 阅读 @tests/base_op 目录的所有文件，理解单元测试的实现逻辑，没有用户的批准，不允许删除和修改原有的单元测试。如果发现有不严谨的单元测试，向用户提出
3. 阅读 @kSimd/kSimd/core/impl/op.hpp 和 @kSimd/kSimd/core/impl/shared.hpp 文件，理解所有的 is_tag_* concept 的含义和各种 Tag 类型的含义
4. 根据用户在对话中的指令，开始编写新的op
5. 修改 @tests/base_op 目录中对应的文件，编写对应的单元测试代码
6. 编写完代码之后，不需要编译运行测试，此时，AI Agent的任务已经完成了，剩下的编译测试就交给用户
