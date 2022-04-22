package xjtuse.castingcurvepredict.data;

import java.util.List;

import org.apache.ibatis.annotations.Param;

public interface TrainTaskMapper {
    public List<TrainTask> getTasks();

    public void createTask(TrainTask task);

    public List<TrainTask> getWorkingTasks();

    public void deleteModelById(int id);

    void createModel(MlModel model);

    public TrainTask getTaskById(long id);

    public void deleteTaskById(int id);
}
