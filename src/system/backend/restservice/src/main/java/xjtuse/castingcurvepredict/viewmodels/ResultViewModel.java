package xjtuse.castingcurvepredict.viewmodels;

import java.util.ArrayList;

import xjtuse.castingcurvepredict.models.CastingResultItemModel;
import xjtuse.castingcurvepredict.models.CastingResultModel;

public class ResultViewModel {
    private ArrayList<String> _times;
    private ArrayList<Double> _values;
    private CastingResultModel _resultModel;

    public ResultViewModel(CastingResultModel resultModel)
    {
        _resultModel = resultModel;
        ArrayList<CastingResultItemModel> results = resultModel.getResultItems();
        ArrayList<String> times = new ArrayList<String>();
        ArrayList<Double> values = new ArrayList<Double>();
        for (CastingResultItemModel resultItem : results) {
            times.add(resultItem.getDatetime());
            values.add(resultItem.getValue());
        }
        _times = times;
        _values = values;
    }

    
    public ArrayList<String> getTimes()
    {
        return _times;
    }

    public ArrayList<Double> getValues()
    {
        return _values;
    }
}
