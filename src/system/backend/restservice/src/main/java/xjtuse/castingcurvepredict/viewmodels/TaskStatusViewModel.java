package xjtuse.castingcurvepredict.viewmodels;

import java.util.Date;
import java.util.Map;

public class TaskStatusViewModel extends StatusViewModel {
    private Map<Date, Double> mLosses;
    private Map<Date, Integer> mEpochs;

    public void setLosses(Map<Date, Double> losses) {
        mLosses = losses;
    }

    public void setEpochs(Map<Date, Integer> epochs) {
        mEpochs = epochs;
    }

    public Map<Date, Double> getLosses() {
        return mLosses;
    }

    public Map<Date, Integer> getEpochs() {
        return mEpochs;
    }

}
