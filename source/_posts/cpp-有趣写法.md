---
title: "C++ const_cast 与成员变量修改"
date: "2024-01-26 09:49:08"
categories: ["C++"]
tags: ["C++", "类型转换"]
description: "记录使用 const_cast 转换 this 指针并修改成员变量的写法。"
permalink: "/2024/01/26/cpp-有趣写法/"
---
不使用this指针操作类变量，通过const限定，但是如果想改变this指针内容，则使用const_cast强转this指针，之后修改。以达到不使用this指针this->修改类变量。
```c++
// Hash code. Any better algorithm?
unsigned int NativeCallStack::hash() const {
  uintptr_t hash_val = _hash_value;
  if (hash_val == 0) {
    for (int index = 0; index < NMT_TrackingStackDepth; index++) {
      if (_stack[index] == NULL) break;
      hash_val += (uintptr_t)_stack[index];
    }

    NativeCallStack* p = const_cast<NativeCallStack*>(this);
    p->_hash_value = (unsigned int)(hash_val & 0xFFFFFFFF);
  }
  return _hash_value;
}
```