---
title: cds
date: 2024-08-13 09:52:05
tags:
---
## CDS
CDS最初是class Data-sharing允许多个JVM通过内存映射的方式节省内存，但后续发现内存不是瓶颈，但是可以节省相同的查找，加载，验证时间。

执行字节码时，JVM需要一些准备的工作。传类名，在磁盘上查找类，加载它，验证字节码，将类装载为自己的内部数据结构，每一步都需要花费一些时间，想想每次JVM都要加载成千上万个类，这时时间上的花费很明显可以看得出来。
因为Jar包并没有改变，class-data一直都是相同的，每次JVM执行的也是相同的查找，加载，验证动作。

CDS由参数控制
> -Xshare:value 其中value可以为以下值
>  * auto : 允许CDS特性在JDK12之后默认位置为: $JAVA_HOME/lib/server/classes.jsa (Windows: $JAVA_HOME/bin/server/classes.jsa)
>  * on : 需要CDS文件，如果JVM在加载CDS文件时遇到问题则会推出并打印错误信息，只应该在测试实验的时候使用。
>  * off : 禁止CDS特性。
>  * dump : 生产CDS文件
## AppCDS
AppCDS相当于在CDS的基础上允许复用部分应用类加载器加载的类对象，他可以配合CDS一起使用。将应用部分的类也可以同时做复用在JDK13和JDK19有enhancements。

使用方式通过参数-XX:ArchiveClassesAtExit=name of archive file指定生成的cds文件位置，之后通过参数-XX:SharedArchiveFile=name of archive file指定archive文件的服用

```shell
// 当开启参数之后默认在对应目录生产dynamic archive文件
java -XX:ArchiveClassesAtExit=petclinic-dynamic-archive.jsa -jar target/spring-petclinic-2.5.1.jar
// 加载指定的dynamic archive文件
java -XX:SharedArchiveFile=petclinic-dynamic-archive.jsa  -jar target/spring-petclinic-2.5.1.jar
```

* static archive

* dynamic archive

```c++
  enum CDSSharedClassFlags {
    _is_shared_class                       = 1 << 0,  // shadows MetaspaceObj::is_shared
    _archived_lambda_proxy_is_available    = 1 << 1,
    _has_value_based_class_annotation      = 1 << 2,
    _verified_at_dump_time                 = 1 << 3,
    _has_archived_enum_objs                = 1 << 4,
    // This class was not loaded from a classfile in the module image
    // or classpath.
    _is_generated_shared_class             = 1 << 5,
    // The archived mirror is already initialized. No need to call <clinit>
    _has_preinitialized_mirror             = 1 << 6,
  };
```