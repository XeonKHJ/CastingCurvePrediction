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
	public static void main(String[] args) throws IOException {
		loadMybatis();
		SpringApplication.run(RestserviceApplication.class, args);
	}

	private static void loadMybatis() throws IOException{
		// load mybatis
		String resource = "mybatis-config.xml";
		InputStream inputStream = Resources.getResourceAsStream(resource);
		mSqlSessionFactory = new SqlSessionFactoryBuilder().build(inputStream);
	}

	public static SqlSessionFactory getSqlSessionFactory()
	{
		return mSqlSessionFactory;
	}
}
