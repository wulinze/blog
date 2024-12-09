---
title: Hottub
date: 2024-07-24 13:28:59
tags:
---
## Hottub学习
* 编译

在adlc.make文件110行增加编译选项
```shell
 -static-libgcc -static-libstdc++
```
在vm.make 311行增加编译选项
```shell
 -Wno-literal-suffix
```


* df2b97cf38ea1c41f3e05d3d9c0a7bb7819a0ed8
```c++
  static PerfVariable*      _begin_vm_creation_time;
  static PerfVariable*      _end_vm_creation_time;
  static PerfVariable*      _vm_init_done_time;
```
private变为public
create_vm 之后打印上述时间
* 