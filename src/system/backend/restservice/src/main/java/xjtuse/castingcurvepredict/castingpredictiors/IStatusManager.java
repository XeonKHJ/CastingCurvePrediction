package xjtuse.castingcurvepredict.castingpredictiors;

import java.util.Date;
import java.util.Map;
import java.util.SortedMap;

import xjtuse.castingcurvepredict.models.TaskModel;

public interface IStatusManager {
    void setTask(TaskModel task);
    TaskModel getTask();
    void saveEpochs(Map<Date, Integer> epochs);
    void saveLosses(Map<Date, Double> losses);
    void saveStatus(TaskStatus status);
    void saveStatus(TaskStatus status, Date time);
    SortedMap<Date, Double> readLosses();
    SortedMap<Date, Integer> readEpochs();
    TaskStatus readStatus();
    void AddEventListener(IStatusManagerEventListener listener);
    void RemoveEventListenr(IStatusManagerEventListener listener);
}
