---
title: CRaC源码分析
date: 2024-11-05 03:23:13
tags:
---
# CRaC 17+1 源码分析
## checkpoint流程

## restore流程

## restore中参数问题
 restore的过程中所有的-D参数可以设定新的值通过add_property的方式对于System.Property设定新的值-XX设定的参数只允许CR相关参数重新制定在参数描述中包含MANAGEABLE或者RESTORE_SETTABLE的参数才可以重新设定。原理在于第二次恢复JVM的过程中，首先会创建一个新的JVM通过shared_memory的方式将可变参数写入，退出的JVM会等待RESTORE_SIGNAL的发送，之后在restore成功后通过信号的方式退出之前的JVM
## criu的调用逻辑
crac-17中调用criu通过实现了内部的criuengine库监控criu的整个执行过程在criu中实现了checkpoint和restore1两个功能