## Table of contents

* [CTE](#common-table-expressions)
* [Set Comparison](#set-comparison)
* [View](#view)
* [Window Functions](#window-functions)
* [Datetime Functions](#datetime-functions)
* [Triggers](#triggers)
* [Database Construction](#database-construction)
* [Sample Questions](#Sample Questions)

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
  select_query
```
## Window Functions
## Datetime Functions
## Triggers
## Database Construction
## Sample Questions
* [HackerRank](https://www.hackerrank.com/domains/sql)
* [LeetCode](https://leetcode.com/problemset/all/?search=sql)
