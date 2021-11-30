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

    public ArrayList<CastingResultItemModel> getResultItems()
    {
        return resultItems;
    }
}
