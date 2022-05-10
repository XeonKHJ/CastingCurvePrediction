package xjtuse.castingcurvepredict.castingpredictiors.dummyimpl;

import java.util.ArrayList;

import xjtuse.castingcurvepredict.castingpredictiors.GeneratorInput;
import xjtuse.castingcurvepredict.castingpredictiors.ICastingGenerator;
import xjtuse.castingcurvepredict.models.CastingModel;
import xjtuse.castingcurvepredict.models.CastingResultModel;
import xjtuse.castingcurvepredict.models.TaskModel;

public class DummyCastingGenerator implements ICastingGenerator {

	private PythonThread _impl;

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
		if(_impl != null)
		{
			_impl.stop();
			_impl = null;
		}

		_impl = new PythonThread(task.getId(), 20);
		_impl.start();
	}

	@Override
	public void stop(TaskModel task) {
		if(_impl != null)
		{
			_impl.stop();
		}
	}

}
