package xjtuse.castingcurvepredict.data;

import java.util.List;

public interface TrainTaskMapper {
    public List<TrainTask> getTasks();

    public void createTask(TrainTask task);

    public List<TrainTask> getWorkingTasks();

    public void deleteModelById(long id);

    void createModel(MlModel model);

    public TrainTask getTaskById(long id);

    public void deleteTaskById(long id);
}
