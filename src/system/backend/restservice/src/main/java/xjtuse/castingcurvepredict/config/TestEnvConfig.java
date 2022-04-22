package xjtuse.castingcurvepredict.config;

import java.util.HashMap;
import java.util.Hashtable;

import xjtuse.castingcurvepredict.castingpredictiors.ICastingGenerator;
import xjtuse.castingcurvepredict.castingpredictiors.IStatusManager;
import xjtuse.castingcurvepredict.castingpredictiors.dummyimpl.DummyCastingGenerator;
import xjtuse.castingcurvepredict.castingpredictiors.dummyimpl.StreamStatusManager;
import xjtuse.castingcurvepredict.castingpredictiors.impls.ConstCastingGenerator;
import xjtuse.castingcurvepredict.models.TaskModel;

public class TestEnvConfig implements IConfigFactory {
    private static Hashtable<Long, IStatusManager> taskStatusMapper = new Hashtable<>();

    @Override
    public ICastingGenerator getCastingGenerator() {
        // TODO Auto-generated method stub
        ICastingGenerator generator = new DummyCastingGenerator();

        return generator;
    }

    @Override
    public String getModelDir() {
        return "C:\\Users\\redal\\source\\repos\\CastingCurvePrediction\\src\\system\\backend\\files\\";
    }

    @Override
    public IStatusManager getStatusManager(TaskModel model) {
        // TODO Auto-generated method stub
        if(taskStatusMapper.get(model.getId()) == null)
        {
            taskStatusMapper.put(model.getId(), createStatusManager());
        }

        return taskStatusMapper.get(model.getId());
    }

    private IStatusManager createStatusManager()
    {
        return new StreamStatusManager();
    }
}
