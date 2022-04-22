package xjtuse.castingcurvepredict.restservice;

import java.io.IOException;
import java.io.InputStream;

import org.apache.ibatis.io.Resources;
import org.apache.ibatis.session.SqlSessionFactory;
import org.apache.ibatis.session.SqlSessionFactoryBuilder;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import xjtuse.castingcurvepredict.config.IConfigFactory;
import xjtuse.castingcurvepredict.config.TestEnvConfig;

@SpringBootApplication
public class RestserviceApplication {
	private static SqlSessionFactory mSqlSessionFactory;
	private static IConfigFactory config = new TestEnvConfig();
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

	public static IConfigFactory getConfig(){
		return config;
	}
}
