package xjtuse.castingcurvepredict.models;

import java.util.ArrayList;
import java.util.List;

public class CastingResultModel {
    
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
