package xjtuse.castingcurvepredict.restservice;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class ServiceTestController {
    @GetMapping("/servicetest")
    public String serviceTest(@RequestParam(value="name", defaultValue="Nobody") String name)
    {
        return String.format("Hello, %s", name);
    }
}
