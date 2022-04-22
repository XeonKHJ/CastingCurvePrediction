package xjtuse.castingcurvepredict.config;

import java.util.HashMap;
import java.util.Hashtable;

import xjtuse.castingcurvepredict.castingpredictiors.ICastingGenerator;
import xjtuse.castingcurvepredict.castingpredictiors.IStatusManager;
import xjtuse.castingcurvepredict.castingpredictiors.IStatusManagerEventListener;
import xjtuse.castingcurvepredict.castingpredictiors.dummyimpl.DummyCastingGenerator;
import xjtuse.castingcurvepredict.castingpredictiors.dummyimpl.StreamStatusManager;
import xjtuse.castingcurvepredict.castingpredictiors.impls.ConstCastingGenerator;
import xjtuse.castingcurvepredict.models.TaskModel;
import xjtuse.castingcurvepredict.viewmodels.IViewModel;

public class TestEnvConfig implements IConfigFactory, IStatusManagerEventListener {
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
        if (taskStatusMapper.get(model.getId()) == null) {
            taskStatusMapper.put(model.getId(), createStatusManager());
        }

        return taskStatusMapper.get(model.getId());
    }

    private IStatusManager createStatusManager() {
        var sm = new StreamStatusManager();
        sm.AddEventListener(this);
        return sm;
    }

    @Override
    public void onTaskStarting(StreamStatusManager statusManager) {
        // TODO Auto-generated method stub

    }

    @Override
    public void onTaskStarted(StreamStatusManager statusManager) {
        // TODO Auto-generated method stub

    }

    @Override
    public void onTaskStopped(StreamStatusManager statusManager) {
        var task = statusManager.getTask();
        if (task != null) {
            taskStatusMapper.remove(task.getId());
        }
    }

    @Override
    public void setErrorMessageOnViewModel(IViewModel viewModel, Exception exception) {
        // TODO Auto-generated method stub
        viewModel.setMessage(exception.getMessage() + exception.getStackTrace().toString());
    }

}
