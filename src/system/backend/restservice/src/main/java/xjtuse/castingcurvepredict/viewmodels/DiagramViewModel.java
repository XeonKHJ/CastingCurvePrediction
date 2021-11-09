package xjtuse.castingcurvepredict.viewmodels;

import xjtuse.castingcurvepredict.models.CastingResultModel;

public class DiagramViewModel {
    private CastingResultModel _resultModel;
    private ResultViewModel _resultViewModel;  
    public DiagramViewModel(CastingResultModel model)
    {   
        _resultModel = model;
        _resultViewModel =  new ResultViewModel(model);
    }
    
    public ResultViewModel getCastingCurveValues()
    {
        return _resultViewModel;
    }
}
