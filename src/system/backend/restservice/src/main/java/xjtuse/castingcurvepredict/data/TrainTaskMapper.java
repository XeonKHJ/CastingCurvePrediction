package xjtuse.castingcurvepredict.data;

import java.util.List;

import org.apache.ibatis.annotations.Param;

public interface TrainTaskMapper {
    public TrainTask getMlModelById(@Param("id") long id);
    public List<TrainTask> getModels();
}
