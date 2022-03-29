package xjtuse.castingcurvepredict.restservice;

import java.io.IOException;
import java.io.InputStream;

import org.apache.ibatis.io.Resources;
import org.apache.ibatis.session.SqlSessionFactory;
import org.apache.ibatis.session.SqlSessionFactoryBuilder;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class RestserviceApplication {
	private static SqlSessionFactory mSqlSessionFactory;
	public static void main(String[] args) {
		SpringApplication.run(RestserviceApplication.class, args);
	}

	public static void loadMybatis() throws IOException{
		// load mybatis
		String resource = "org/mybatis/example/mybatis-config.xml";
		InputStream inputStream = Resources.getResourceAsStream(resource);
		mSqlSessionFactory = new SqlSessionFactoryBuilder().build(inputStream);
	}

	public static SqlSessionFactory getSqlSessionFactory()
	{
		return mSqlSessionFactory;
	}
}
