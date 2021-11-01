package xjtuse.castingcurvepredict.castingpredictiors;

import xjtuse.castingcurvepredict.interfaces.*;
import xjtuse.castingcurvepredict.models.CastingInputModel;
import xjtuse.castingcurvepredict.models.CastingResultModel;

public class ConstCastingGenerator implements ICastingGenerator {
    public ConstCastingGenerator()
    {
        
    }

    @Override
    public CastingResultModel PredcitCastingCurve(CastingInputModel input) {
        // Read data from json
        CastingResultModel resultModel = new CastingResultModel();

        

        return resultModel;
    }

}
