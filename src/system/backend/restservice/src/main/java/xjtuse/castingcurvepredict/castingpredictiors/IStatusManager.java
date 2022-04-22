package xjtuse.castingcurvepredict.castingpredictiors;

import java.util.Date;
import java.util.Map;
import java.util.SortedMap;

public interface IStatusManager {
    void saveEpochs(Map<Date, Integer> epochs);
    void saveLosses(Map<Date, Double> losses);
    void saveStatus(TaskStatus status);
    void saveStatus(TaskStatus status, Date time);
    SortedMap<Date, Double> readLosses();
    SortedMap<Date, Integer> readEpochs();
    TaskStatus readStatus();
}
