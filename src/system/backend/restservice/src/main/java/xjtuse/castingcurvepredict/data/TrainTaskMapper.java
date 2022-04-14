package xjtuse.castingcurvepredict.data;

import java.util.List;

import org.apache.ibatis.annotations.Param;

public interface TrainTaskMapper {
    public TrainTask getMlModelById(@Param("id") long id);

    public List<TrainTask> getModels();

    public void createTask(TrainTask task);

    public List<TrainTask> getWorkingTasks();

    public void deleteModelById(int id);

    void createModel(MlModel model);

    public MlModel getModelById(int id);

    public void deleteTaskById(int id);
}
