import java.io.*;
import java.util.*;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapred.*;
import org.apache.hadoop.util.*;


public class exercise3 extends Configured implements Tool {         
	public static class Map extends MapReduceBase implements Mapper<LongWritable, Text, Text, DoubleWritable> {
		public void map(LongWritable key, Text value, OutputCollector<Text, DoubleWritable> output, Reporter reporter) throws IOException {
		    String line = value.toString();
            String[] values = line.split(",");

            String title = values[0].trim();
            String artist = values[2].trim();
            Double duration = Double.parseDouble(values[3].trim()); 

            String yearStr = values[165].trim();
            int yearInt=0;
            boolean yearIsInt = true;
            try{
                yearInt = Integer.parseInt(yearStr);
            } catch(NumberFormatException e){  
                yearIsInt = false;
            } 

        	if (yearIsInt&&(yearInt<=2010)&&(yearInt>=2000)){
                output.collect(new Text(title+","+artist), new DoubleWritable(duration));
            }
        }
	}

	public int run(String[] args) throws Exception {
		JobConf conf = new JobConf(getConf(), exercise3.class);
        conf.set("mapred.textoutputformat.separator", ",");
		conf.setJobName("hw1_exercise3_lf");

		conf.setNumReduceTasks(0);

        conf.setMapOutputKeyClass(Text.class);
	    conf.setMapOutputValueClass(DoubleWritable.class);
		conf.setOutputKeyClass(Text.class);
		conf.setOutputValueClass(DoubleWritable.class);

        conf.setMapperClass(Map.class);

		conf.setInputFormat(TextInputFormat.class);
		conf.setOutputFormat(TextOutputFormat.class);

		FileInputFormat.setInputPaths(conf, new Path(args[0]));
		FileOutputFormat.setOutputPath(conf, new Path(args[1]));

		JobClient.runJob(conf);
		return 0;
    }
    public static void main(String[] args) throws Exception {
		int res = ToolRunner.run(new Configuration(), new exercise3(), args);
		System.exit(res);
    }
}