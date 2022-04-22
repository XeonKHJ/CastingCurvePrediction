package xjtuse.castingcurvepredict.castingpredictiors.dummyimpl;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;

import xjtuse.castingcurvepredict.castingpredictiors.GeneratorInput;
import xjtuse.castingcurvepredict.castingpredictiors.ICastingGenerator;
import xjtuse.castingcurvepredict.models.CastingModel;
import xjtuse.castingcurvepredict.models.CastingResultModel;

public class DummyCastingGenerator implements ICastingGenerator  {

    @Override
    public CastingResultModel PredcitCastingCurve(GeneratorInput input) {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public void updateModel(CastingModel data) {
        // TODO Auto-generated method stub
        
    }

    @Override
    public void updateModel(ArrayList<CastingModel> datas) {
        // TODO Auto-generated method stub
        
    }

    @Override
    public void train() {
        // TODO 训练模型
        try {
			Runtime rt = Runtime.getRuntime();
			//javac后无具体的参数，会输出错误信息。
			Process p = rt.exec("javac");
			//获取错误信息流。
			InputStream stderr = p.getErrorStream();
			//将错误信息流输出
			InputStreamReader isr = new InputStreamReader(stderr);
			BufferedReader br = new BufferedReader(isr);
			String line = "";
			System.out.println("--------------error---------------");
			while((line = br.readLine()) != null)
				System.out.println(line);
			System.out.println("");
			//等待进程完成。
			int exitVal = p.waitFor();
			System.out.println("Process Exitvalue: " + exitVal);
		}catch(Throwable t) {
			t.printStackTrace();
		}
    }
    
}
