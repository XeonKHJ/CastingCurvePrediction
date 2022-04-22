package xjtuse.castingcurvepredict.config;

import xjtuse.castingcurvepredict.castingpredictiors.ICastingGenerator;
import xjtuse.castingcurvepredict.castingpredictiors.IStatusManager;
import xjtuse.castingcurvepredict.models.TaskModel;
import xjtuse.castingcurvepredict.viewmodels.IViewModel;

public interface IConfigFactory {
    ICastingGenerator getCastingGenerator();
    String getModelDir();
    IStatusManager getStatusManager(TaskModel model);
    void setErrorMessageOnViewModel(IViewModel viewModel, Exception exception);
}
