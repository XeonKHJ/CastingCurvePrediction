package xjtuse.castingcurvepredict.interfaces;

import java.util.ArrayList;

import xjtuse.castingcurvepredict.models.CastingInputModel;
import xjtuse.castingcurvepredict.models.CastingModel;
import xjtuse.castingcurvepredict.models.CastingResultModel;
import xjtuse.castingcurvepredict.models.PredictionModel;

public interface ICastingGenerator {
    public CastingResultModel PredcitCastingCurve(CastingInputModel input);
    
    public ArrayList<PredictionModel> getModelList();

    public void updateModel(CastingModel data);
    public void updateModel(ArrayList<CastingModel> datas);
}
