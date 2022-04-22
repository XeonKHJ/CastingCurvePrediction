package xjtuse.castingcurvepredict.models;

import xjtuse.castingcurvepredict.castingpredictiors.GeneratorInput;
import xjtuse.castingcurvepredict.castingpredictiors.ICastingGenerator;

public class CastingModel {
    private ICastingGenerator _generator;

    public CastingModel(ICastingGenerator generator) {
        _generator = generator;
    }

    public CastingResultModel PredictCastingCurve(GeneratorInput inputModel) {
        return _generator.PredcitCastingCurve(inputModel);
    }
}
