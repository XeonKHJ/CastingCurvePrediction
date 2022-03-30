package xjtuse.castingcurvepredict.viewmodels;
import xjtuse.castingcurvepredict.models.CastingResultModel;

public class DiagramViewModel extends StatusViewModel {
    private CastingResultModel _resultModel;
    private ResultViewModel _resultViewModel;  
    public DiagramViewModel(CastingResultModel model)
    {   
        _resultModel = model;
        _resultViewModel =  new ResultViewModel(model);
        setStatusCode(model.getStatusCode());
        setMessage(model.getMessage());
    }
    
    public ResultViewModel getCastingCurveValues()
    {
        return _resultViewModel;
    }

    public CastingResultModel getResultModel()
    {
        return _resultModel;
    }
}
