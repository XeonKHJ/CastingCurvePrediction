package xjtuse.castingcurvepredict.config;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Hashtable;
import org.apache.ibatis.session.SqlSession;

import xjtuse.castingcurvepredict.castingpredictiors.ICastingGenerator;
import xjtuse.castingcurvepredict.castingpredictiors.IStatusManager;
import xjtuse.castingcurvepredict.castingpredictiors.IStatusManagerEventListener;

import xjtuse.castingcurvepredict.castingpredictiors.dummyimpl.DummyCastingGenerator;
import xjtuse.castingcurvepredict.castingpredictiors.dummyimpl.StreamStatusManager;

import xjtuse.castingcurvepredict.data.MlModelMapper;
import xjtuse.castingcurvepredict.models.TaskModel;
import xjtuse.castingcurvepredict.restservice.RestserviceApplication;
import xjtuse.castingcurvepredict.viewmodels.IViewModel;

public class TestEnvConfig implements IConfigFactory, IStatusManagerEventListener {
    private static Hashtable<Long, IStatusManager> taskStatusMapper = new Hashtable<>();

    @Override
    public ICastingGenerator getCastingGenerator() {
        ICastingGenerator generator = new DummyCastingGenerator();
        return generator;
    }

    @Override
    public String getModelDir() {
        return "C:\\Users\\redal\\source\\repos\\CastingCurvePrediction\\src\\system\\backend\\files\\";
    }

    @Override
    public IStatusManager getStatusManager(TaskModel model) {
        if (taskStatusMapper.get(model.getId()) == null) {
            var sm = createStatusManager();
            sm.setTask(model);
            taskStatusMapper.put(model.getId(), sm);
        }

        return taskStatusMapper.get(model.getId());
    }

    private IStatusManager createStatusManager() {
        var sm = new StreamStatusManager();
        sm.AddEventListener(this);
        return sm;
    }

    @Override
    public void setErrorMessageOnViewModel(IViewModel viewModel, Exception exception) {
        // TODO 给ViewModel设置错误信息
        viewModel.setMessage(exception.getMessage() + exception.getStackTrace().toString());
    }


    @Override
    public IStatusManager getStatusManager(long taskId) {
        if (taskStatusMapper.get(taskId) == null) {
            // TODO 当taskid不存在时应该怎么处理
        }

        return taskStatusMapper.get(taskId);
    }

    @Override
    public long generateTaskId() throws IndexOutOfBoundsException {
        long avaliableId = -1;
        for (avaliableId = 0; avaliableId < Long.MAX_VALUE; ++avaliableId) {
            if (taskStatusMapper.get(avaliableId) == null) {
                break;
            }
        }

        if (avaliableId == Long.MAX_VALUE) {
            throw new IndexOutOfBoundsException("任务数量达到上限");
        }

        return avaliableId;
    }

    @Override
    public Collection<TaskModel> getTasks() {
        ArrayList<TaskModel> tasks = new ArrayList<>();
        for (var key : taskStatusMapper.keySet()) {
            tasks.add(taskStatusMapper.get(key).getTask());
        }

        return tasks;
    }

    @Override
    public void onTaskStarting(IStatusManager statusManager) {
        // TODO 任务开始时触发，更新数据库中学习模型的状体。

    }

    @Override
    public void onTaskStarted(IStatusManager statusManager) {
        var sessionFactory = RestserviceApplication.getSqlSessionFactory();
        
        try (SqlSession session = sessionFactory.openSession()) {
            var mlModelMapper = session.getMapper(MlModelMapper.class);
            mlModelMapper.UpdateMlModelStatusById(statusManager.getTask().getModelId(), "Training");
            session.commit();
        } catch (Exception ex) {
            System.out.println(ex.getMessage());
        }
    }

    @Override
    public void onTaskStopped(IStatusManager statusManager) {
        var task = statusManager.getTask();
        if (task != null) {
            taskStatusMapper.remove(task.getId());
        }

        var sessionFactory = RestserviceApplication.getSqlSessionFactory();
        try (SqlSession session = sessionFactory.openSession()) {
            var mlModelMapper = session.getMapper(MlModelMapper.class);
            mlModelMapper.UpdateMlModelLossById(statusManager.getTask().getModelId(), task.getLoss());
            session.commit();
        } catch (Exception ex) {
            System.out.println(ex.getMessage());
        }
    }

    
    @Override
    public void onTaskCompleted(IStatusManager statusManager) {
        var task = statusManager.getTask();
        task.Stop();
        var sessionFactory = RestserviceApplication.getSqlSessionFactory();

        try (SqlSession session = sessionFactory.openSession()) {
            var mlModelMapper = session.getMapper(MlModelMapper.class);
            mlModelMapper.UpdateMlModelStatusById(task.getModelId(), "trained");
        } catch (Exception ex) {
            System.out.println(ex.getMessage());
        }
    }

    @Override
    public TaskModel getTaskFromModelId(long id) {
        TaskModel result = null;
        for (var key : taskStatusMapper.keySet()) {
            if(id == taskStatusMapper.get(key).getTask().getModelId())
            {
                result = taskStatusMapper.get(key).getTask();
            }
        }
        return result;
    }
}
