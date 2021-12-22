package xjtuse.castingcurvepredict.viewmodels;

import java.util.ArrayList;

import xjtuse.castingcurvepredict.models.CastingResultItemModel;
import xjtuse.castingcurvepredict.models.CastingResultModel;

public class ResultViewModel extends StatusViewModel {
    private ArrayList<String> _times;
    private ArrayList<Double> _values;
    private ArrayList<Double> _liqLevels;
    private CastingResultModel _resultModel;

    public ResultViewModel(CastingResultModel resultModel)
    {
        _resultModel = resultModel;
        ArrayList<CastingResultItemModel> results = resultModel.getResultItems();
        ArrayList<String> times = new ArrayList<String>();
        ArrayList<Double> values = new ArrayList<Double>();
        ArrayList<Double> liqLevels = new ArrayList<Double>();
        for (CastingResultItemModel resultItem : results) {
            times.add(resultItem.getDatetime());
            values.add(resultItem.getStopperPos());
            liqLevels.add(resultItem.getLiqLevel());
        }
        _times = times;
        _values = values;

        _liqLevels = liqLevels;
    }

    
    public ArrayList<String> getTimes()
    {
        return _times;
    }

    public ArrayList<Double> getValues()
    {
        return _values;
    }

    public ArrayList<Double> getLiqLevel()
    {
        return _liqLevels;
    }
}
