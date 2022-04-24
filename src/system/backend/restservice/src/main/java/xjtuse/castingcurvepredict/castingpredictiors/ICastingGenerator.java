package xjtuse.castingcurvepredict.castingpredictiors;

import java.util.ArrayList;

import xjtuse.castingcurvepredict.models.CastingModel;
import xjtuse.castingcurvepredict.models.CastingResultModel;
import xjtuse.castingcurvepredict.models.TaskModel;

public interface ICastingGenerator {
    // 做预测
    CastingResultModel PredcitCastingCurve(GeneratorInput input);
    // 更新模型
    void updateModel(CastingModel data);

    void updateModel(ArrayList<CastingModel> datas);

    void train(TaskModel task);
}
