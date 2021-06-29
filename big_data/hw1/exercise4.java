import java.io.*;
import java.util.*;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapred.*;
import org.apache.hadoop.util.*;


public class exercise4 extends Configured implements Tool {
	public static class Map extends MapReduceBase implements Mapper<LongWritable, Text, Text, DoubleWritable> {
		public void map(LongWritable key, Text value, OutputCollector<Text, DoubleWritable> output, Reporter reporter) throws IOException {
		    String line = value.toString();
            String[] values = line.split(",");

            String artist = values[2];
            Double duration = Double.parseDouble(values[3]);

        	output.collect(new Text(artist), new DoubleWritable(duration));
        }
	}

    public static class Partition implements Partitioner<Text, DoubleWritable> {
        public void configure(JobConf job) {
        }

        public int getPartition(Text key, DoubleWritable value, int numReduceTasks) {
            String artist = key.toString();
        
            if(numReduceTasks == 0) {
                return 0;
            }
                
            char start = artist.charAt(0);
            if(start<='e') {
                return 0; 
            } else if (start>'e' && start<='j'){
                return 1;
            } else if (start>'j' && start<='o'){
                return 2;
            } else if (start>'o' && start<='t'){
                return 3;
            } else { 
                return 4;
            }
        }
    }

	public static class Reduce extends MapReduceBase implements Reducer<Text, DoubleWritable, Text, DoubleWritable> {          
		public void reduce(Text key, Iterator<DoubleWritable> values, OutputCollector<Text, DoubleWritable> output, Reporter reporter) throws IOException {
            Double maxDuration = 0.0;

			while (values.hasNext()) {
                Double curDuration = values.next().get();
                
                if (curDuration>=maxDuration) {
                    maxDuration = curDuration;
                }
            }
            
			output.collect(key, new DoubleWritable(maxDuration));
		}
	}

	public int run(String[] args) throws Exception {
		JobConf conf = new JobConf(getConf(), exercise4.class);
        conf.set("mapred.textoutputformat.separator", ",");
		conf.setJobName("hw1_exercise4_lf");

		conf.setNumReduceTasks(5);

        conf.setMapOutputKeyClass(Text.class);
	    conf.setMapOutputValueClass(DoubleWritable.class);
		conf.setOutputKeyClass(Text.class);
		conf.setOutputValueClass(DoubleWritable.class);

        conf.setMapperClass(Map.class);
        conf.setPartitionerClass(Partition.class);
		conf.setReducerClass(Reduce.class);

		conf.setInputFormat(TextInputFormat.class);
		conf.setOutputFormat(TextOutputFormat.class);

		FileInputFormat.setInputPaths(conf, new Path(args[0]));
		FileOutputFormat.setOutputPath(conf, new Path(args[1]));

		JobClient.runJob(conf);
		return 0;
    }

    public static void main(String[] args) throws Exception {
		int res = ToolRunner.run(new Configuration(), new exercise4(), args);
		System.exit(res);
    }
}