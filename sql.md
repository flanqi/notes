## Table of contents

* [CTE](#common-table-expressions)
* [Set Comparison](#set-comparison)
* [View](#view)
* [Window Functions](#window-functions)
* [Datetime Functions](#datetime-functions)
* [Triggers](#triggers)
* [Database Construction](#database-construction)
* [Sample Questions](#sample-questions)

## Common Table Expressions
```sql
with table_name(col_list) as
(
select ... from ...
)
select * from table_name;
```
### Recursive CTE
```sql
with recursive table_name(col_list) as
(
initial_select_statement

union (all)

recursive_select
)
select * from table_name;
```
**note:** mysql prior to 8.0 does not support with clause
## Set Comparison
### SOME / ANY
Find the instructors having salary more than at least one of the instructor in the biology department.
```sql
select name from instructors
where salary > some (
  select salary from instructors where department = 'biology'
)
```
### ALL
Find the instructors having salary more than all the instructor in the biology department.
```sql
select name from instructors
where salary > all (
  select salary from instructors where department = 'biology'
)
```

### EXISTS
```sql
select name from employee as e
where exists (select salary from manager as m
              where e.salary > m.salary);
```
NOT EXISTS does exactly the opposite.

## View
Creating view saves time when some query is often used. It creates a "screenshot" for some table.
```sql
create view tmp as
  select_query;
```
## Window Functions
### ROWS / RANGE / GROUPS
sum over values of 3 preceding rows and the current row after ordering
```sql
select sum(quantity) over (order by quantity rows 3 preceding) as rowSum 
from items;
```
sum over values from C - 3 to C if current row value is C
```sql
select sum(quantity) over (order by quantity range 3 preceding) as rangeSum
from items;
```
sum over values from 3 preceding groups and the current group
```sql
select sum(quantity) over (order by quantity groups 3 preceding) as groupSum
from items;
```
**Note:** Instead of N PRECEDING, one can use UNBOUNDED PRECEDING/FOLLOWING, or use rows/range/groups BETWEEN 1 PRECEDING AND 1 FOLLOWING.

### PARTITION BY
```sql
select rank() over (partition by month order by quantity desc) as sales_monthly_rank from items;
```
**other window functions**

DENSE_RANK(): rank with no gaps 

ROW_NUMBER()


## Datetime Functions
### DATEDIFF
Returns the number of dates between two dates
```sql
select datediff("2017-06-25", "2017-06-15");
```
## Triggers
## Database Construction
## Sample Questions
* [HackerRank](https://www.hackerrank.com/domains/sql)
* [LeetCode](https://leetcode.com/problemset/all/?search=sql)
