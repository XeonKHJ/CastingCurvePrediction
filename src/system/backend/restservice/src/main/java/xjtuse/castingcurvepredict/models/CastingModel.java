package xjtuse.castingcurvepredict.models;

import xjtuse.castingcurvepredict.interfaces.ICastingGenerator;

public class CastingModel {
    private ICastingGenerator _generator;

    public CastingModel(ICastingGenerator generator) {
        _generator = generator;
    }

    public CastingResultModel PredictCastingCurve(CastingInputModel inputModel) {
        return _generator.PredcitCastingCurve(inputModel);
    }
}
