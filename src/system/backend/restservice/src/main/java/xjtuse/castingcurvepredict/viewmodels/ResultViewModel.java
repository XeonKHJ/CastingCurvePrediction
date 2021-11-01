package xjtuse.castingcurvepredict.viewmodels;

import java.util.ArrayList;

import xjtuse.castingcurvepredict.models.CastingResultItemModel;
import xjtuse.castingcurvepredict.models.CastingResultModel;

public class ResultViewModel {
    private ArrayList<String> _dates;
    private ArrayList<Double> _values;
    private CastingResultModel _resultModel;

    public ResultViewModel(CastingResultModel resultModel)
    {
        _resultModel = resultModel;
        ArrayList<CastingResultItemModel> results = resultModel.getResultItems();
        ArrayList<String> dates = new ArrayList<String>();
        ArrayList<Double> values = new ArrayList<Double>();
        for (CastingResultItemModel resultItem : results) {
            dates.add(resultItem.getDatetime());
            values.add(resultItem.getValue());
        }
        _dates = dates;
        _values = values;
    }

    
    public ArrayList<String> getDates()
    {
        return _dates;
    }

    public ArrayList<Double> getValues()
    {
        return _values;
    }
}
