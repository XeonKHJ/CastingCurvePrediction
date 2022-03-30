自行在 ./src/main/resources 中添加一个mybatis-config.xml的文件。内容如下。

```
<?xml version="1.0" encoding="UTF-8" ?>
<!DOCTYPE configuration
  PUBLIC "-//mybatis.org//DTD Config 3.0//EN"
  "http://mybatis.org/dtd/mybatis-3-config.dtd">

<configuration>
<typeAliases>
  <package name="xjtuse.castingcurvepredict.data"/>
</typeAliases>
  <environments default="development">
    <environment id="development">
      <transactionManager type="JDBC"/>
      <dataSource type="POOLED">
        <property name="driver" value="com.mysql.cj.jdbc.Driver"/>
        <property name="url" value="jdbc:mysql://address=(protocol=tcp)(host=::1)(port=3306)/castingcurvedb"/>
        <property name="username" value="{username}"/>
        <property name="password" value="{abcd}"/>
      </dataSource>
    </environment>
  </environments>
  <mappers>
    <mapper resource="xjtuse/castingcurvepredict/data/MlModelMapper.xml"/>
  </mappers>
</configuration>
```