package xjtuse.castingcurvepredict.models;

import java.util.ArrayList;

import xjtuse.castingcurvepredict.viewmodels.StatusViewModel;

public class CastingResultModel extends StatusViewModel {
    
    ArrayList<CastingResultItemModel> resultItems = new ArrayList<CastingResultItemModel>();
    public CastingResultModel()
    {
        
    }

    public void addResultItem(String time, double value)
    {
        CastingResultItemModel resultItem = new CastingResultItemModel(time, value);
        resultItems.add(resultItem);
    }

    public void addResultItem(String time, double stdPos, double liqLevel)
    {
        CastingResultItemModel resultItem = new CastingResultItemModel(time, stdPos, liqLevel, 0, 0);
        resultItems.add(resultItem);
    }

    public void addResultItem(String time, double stdPos, double liqLevel, double tudishWeight, double ladleWeight)
    {
        CastingResultItemModel resultItem = new CastingResultItemModel(time, stdPos, liqLevel, tudishWeight, ladleWeight);
        resultItems.add(resultItem);
    }

    public ArrayList<CastingResultItemModel> getResultItems()
    {
        return resultItems;
    }
}
