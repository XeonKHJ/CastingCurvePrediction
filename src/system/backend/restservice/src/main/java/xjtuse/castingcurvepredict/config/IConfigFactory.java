package xjtuse.castingcurvepredict.config;

import java.util.Collection;

import xjtuse.castingcurvepredict.castingpredictiors.ICastingGenerator;
import xjtuse.castingcurvepredict.castingpredictiors.IStatusManager;
import xjtuse.castingcurvepredict.models.TaskModel;
import xjtuse.castingcurvepredict.viewmodels.IViewModel;

public interface IConfigFactory {
    ICastingGenerator getCastingGenerator();
    String getModelDir();
    IStatusManager getStatusManager(TaskModel model);
    IStatusManager getStatusManager(long taskId);
    long generateTaskId() throws IndexOutOfBoundsException;
    TaskModel getTaskFromModelId(long id);
    void setErrorMessageOnViewModel(IViewModel viewModel, Exception exception);
    Collection<TaskModel> getTasks();
}
