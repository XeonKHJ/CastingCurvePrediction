自行在 ./src/main/resources 中添加一个mybatis-extra-config.properties的文件。内容如下。

```
driver = com.mysql.cj.jdbc.Driver
url = jdbc:mysql://address=(protocol=tcp)(host=::1)(port=3306)/castingcurvedb
username = database_username
password = database_password
```