package xjtuse.castingcurvepredict.config;

import xjtuse.castingcurvepredict.castingpredictiors.ICastingGenerator;
import xjtuse.castingcurvepredict.castingpredictiors.IStatusManager;
import xjtuse.castingcurvepredict.models.TaskModel;

public interface IConfigFactory {
    ICastingGenerator getCastingGenerator();
    String getModelDir();
    IStatusManager getStatusManager(TaskModel model);
}
