package xjtuse.castingcurvepredict.viewmodels;

import java.util.ArrayList;

import xjtuse.castingcurvepredict.models.CastingResultItemModel;
import xjtuse.castingcurvepredict.models.CastingResultModel;

public class ResultViewModel extends StatusViewModel {
    private ArrayList<String> _times;
    private ArrayList<Double> _values;
    private ArrayList<Double> _liqLevels;
    private ArrayList<Double> _tudishWeights;
    private ArrayList<Double> _ladleWeights;
    private CastingResultModel _resultModel;

    public ResultViewModel(CastingResultModel resultModel)
    {
        _resultModel = resultModel;
        ArrayList<CastingResultItemModel> results = resultModel.getResultItems();
        ArrayList<String> times = new ArrayList<String>();
        ArrayList<Double> values = new ArrayList<Double>();
        ArrayList<Double> liqLevels = new ArrayList<Double>();
        _tudishWeights = new ArrayList<>();
        _ladleWeights = new ArrayList<>();
        for (CastingResultItemModel resultItem : results) {
            times.add(resultItem.getDatetime());
            values.add(resultItem.getStopperPos());
            liqLevels.add(resultItem.getLiqLevel());
            _tudishWeights.add(resultItem.getTudishWeight());
            _ladleWeights.add(resultItem.getLadleWeight());
        }
        _times = times;
        _values = values;

        _liqLevels = liqLevels;
    }

    
    public ArrayList<String> getTimes()
    {
        return _times;
    }

    public ArrayList<Double> getStpPos()
    {
        return _values;
    }

    public ArrayList<Double> getLiqLevel()
    {
        return _liqLevels;
    }

    public ArrayList<Double> getTudishWeights()
    {
        return _tudishWeights;
    }

    public ArrayList<Double> getLadleWeights()
    {
        return _ladleWeights;
    }
}
