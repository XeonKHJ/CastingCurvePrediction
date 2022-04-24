package xjtuse.castingcurvepredict.castingpredictiors.dummyimpl;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;

import xjtuse.castingcurvepredict.castingpredictiors.GeneratorInput;
import xjtuse.castingcurvepredict.castingpredictiors.ICastingGenerator;
import xjtuse.castingcurvepredict.models.CastingModel;
import xjtuse.castingcurvepredict.models.CastingResultModel;
import xjtuse.castingcurvepredict.models.TaskModel;

public class DummyCastingGenerator implements ICastingGenerator {

	private static String condaEnv = "C:\\Users\\redal\\source\\tools\\miniconda3";
	private static String activateCondaCmd = "cmd.exe \"/K\" C:\\Users\\redal\\source\\tools\\miniconda3\\Scripts\\activate.bat C:\\Users\\redal\\source\\tools\\miniconda3";

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
	public void train(TaskModel task) {
		// TODO 训练模型
		PythonThread runable = new PythonThread();
		runable.start();
	}

}
