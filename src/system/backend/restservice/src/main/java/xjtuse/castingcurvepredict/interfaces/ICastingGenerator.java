package xjtuse.castingcurvepredict.interfaces;

import xjtuse.castingcurvepredict.models.CastingInputModel;
import xjtuse.castingcurvepredict.models.CastingResultModel;

public interface ICastingGenerator {
    public CastingResultModel PredcitCastingCurve(CastingInputModel input);
}
