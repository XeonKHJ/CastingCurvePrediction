package xjtuse.castingcurvepredict.viewmodels;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.Map;

import xjtuse.castingcurvepredict.utils.utils;

public class TaskStatusViewModel extends StatusViewModel {
    private ArrayList<String> mLossDates = new ArrayList<>();
    private ArrayList<Double> mLosses = new ArrayList<>();
    private Map<Date, Integer> mEpochs;

    public void setLosses(Map<Date, Double> losses) {
        for(var d : losses.entrySet())
        {
            mLossDates.add(utils.dateToString(d.getKey()));
            mLosses.add(d.getValue());
        }
    }

    public void setEpochs(Map<Date, Integer> epochs) {
        mEpochs = epochs;
    }

    public ArrayList<String> getLossDates(){
        return mLossDates;
    }

    public ArrayList<Double> getLosses() {
        return mLosses;
    }

    public Map<Date, Integer> getEpochs() {
        return mEpochs;
    }

}
