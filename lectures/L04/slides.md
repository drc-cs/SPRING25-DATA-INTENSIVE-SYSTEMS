---
title: MBAI
separator: <!--s-->
verticalSeparator: <!--v-->
theme: serif
revealOptions:
  transition: 'none'
---

<div class = "col-wrapper">
  <div class="c1 col-centered">
  <div style="font-size: 0.8em; left: 0; width: 50%; position: absolute;">

  # Data Intensive Systems
  ## L.04

  </div>
  </div>
  <div class="c2 col-centered" style = "bottom: 0; right: 0; width: 100%; padding-top: 10%">

  <iframe src="https://lottie.host/embed/216f7dd1-8085-4fd6-8511-8538a27cfb4a/PghzHsvgN5.lottie" height = "100%" width = "100%"></iframe>
  </div>
</div>

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
  <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Welcome to Data Intensive Systems.
  ## Please check in by creating an account and entering the code on the chalkboard.

  </div>
  </div>
  <div class="c2 col-centered" style = "bottom: 0; right: 0; width: 40%; padding-top: 5%">
    <iframe src = "https://drc-cs-9a3f6.firebaseapp.com/?label=Check In" width = "100%" height = "100%"></iframe>
  </div>
</div>

<!--s-->

## Announcements

- 

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Intro Poll
  ## On a scale of 1-5, how confident are you with the **OLAP** concepts?

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Intro Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->

<div class="header-slide">

# L.10 | Online Analytical Processing (OLAP)

</div>

<!--s-->

## Scenario: Upgrading from PostgreSQL to a Columnar Database

#### Background
Acme Corp, a leading e-commerce company, has been using PostgreSQL as their primary database for transactional operations. The database handles customer orders, inventory management, and other day-to-day operations efficiently. However, as the company grows, the need for advanced analytics and real-time business intelligence has become critical.

#### Challenge
The current PostgreSQL setup is struggling to keep up with the increasing volume of data and the complexity of analytical queries. Reports that used to take minutes are now taking hours, and the performance of the transactional system is being impacted by the heavy read operations required for analytics.

#### Proposed Solution
Acme Corp should migrate their analytical workloads to a columnar database such as Amazon Redshift, Google BigQuery, or Snowflake. This will allow them to maintain their PostgreSQL setup for transactional operations while leveraging the power of columnar storage for analytics.

By upgrading to a columnar database, Acme Corp will be well-positioned to meet their growing analytical needs and maintain a competitive edge in the market.

<!--s-->

## Agenda

1. What is Online Analytical Processing (OLAP)?
    - Online Transactional Processing (OLTP) Recap
    - OLTP vs OLAP
    - OLAP Solutions
2. Rise and Fall of the OLAP Cube
3. Columnar Databases
4. Modern Data Science

<!--s-->

## What is OLTP?

**Online Transaction Processing (OLTP)** is a class of software applications capable of supporting transaction-oriented programs. OLTP systems are designed to manage a large number of short online transactions (INSERT, UPDATE, DELETE).

JOIN operations are common in OLTP systems, however, they are expensive.

<div class = "col-wrapper">
  <div class="c1">

  ### Characteristics of OLTP:

  - High transaction volume
  - Short response time
  - Data integrity
  - Normalized data

  </div>
  <div class="c2" style="width: 50%; height: auto;">
  <img style="border-radius: 10px;" src="https://planetscale.com/assets/blog/content/schema-design-101-relational-databases/db72cc3ac506bec544588454972113c4dc3abe50-1953x1576.png" />
<p style="text-align: center; font-size: 0.6em; color: grey;">Ramos 2022</p>

  </div>
</div>

<!--s-->

## OLTP vs OLAP

<div class = "col-wrapper">
<div class="c1" style="width: 50%; height: auto;">

### Online **Transaction** Processing (OLTP)
OLTP is designed for managing transaction-oriented applications.

</div>

<div class="c2" style="width: 50%; height: auto;">

### Online **Analytical** Processing (OLAP)

OLAP is designed for data analysis and decision-making.

</div>
</div>

<!--s-->

## OLTP vs OLAP

| Feature | OLTP | OLAP |
|---------|------|------|
| Purpose | Transaction processing | Data analysis |
| Data Model | Usually Normalized | Usually Denormalized |
| Queries | Simple, short | Complex, long |
| Response Time | Short | Long |
| Data Updates | Frequent | Infrequent |

<!--s-->

## What is OLAP?

OLAP is an approach to answer multi-dimensional analytical queries swiftly. OLAP allows analysts, managers, and executives to gain insight through rapid, consistent, and interactive access to a wide variety of possible views of data.

<div class = "col-wrapper">
<div class="c1" style="width: 50%; height: auto;">

### Characteristics of OLAP:

- Designed for complex queries
- Read-heavy workloads

</div>

<div class="c2" style="width: 50%; height: auto;">

<img style="border-radius: 10px;" src="https://i0.wp.com/olap.com/wp-content/uploads/2019/06/olap-3d-cube.png?fit=2016%2C1890&ssl=1" />
<p style="text-align: center; font-size: 0.6em; color: grey;">olap.com 2025</p>

</div>
</div>

<!--s-->

## Database Schemas

A database schema is the skeleton structure that represents the logical view of the entire database. It defines how data is organized and how relationships between data are handled. Importantly, the schema of a database has consequences for performance in different workloads.

For example, a schema optimized for OLTP (e.g. normalized) will not perform well for OLAP workloads which are read-heavy. Normalized schemas are designed to minimize redundancy and ensure data integrity, but they can lead to complex queries that require multiple joins, which can be slow for analytical workloads.

<img src = "https://cdn-ajfbi.nitrocdn.com/GuYcnotRkcKfJXshTEEKnCZTOtUwxDnm/assets/images/optimized/rev-c2378d8/www.astera.com/wp-content/uploads/2024/05/Database-schema.png" style="border-radius: 10px"/>
<p style="text-align: center; font-size: 0.6em; color: grey;">Asteria 2024</p>

<!--s-->

## Database Schemas

The Star Schema is a type of database schema that is optimized for data warehousing and OLAP applications. It consists of a central fact table surrounded by dimension tables. Star Schemas are typically denormalized, meaning that they contain redundant data to optimize read performance.

<div class="col-wrapper col-centered">
  <img src="https://cdn.prod.website-files.com/5e6f9b297ef3941db2593ba1/614df58a1f10f92b88f95709_Screenshot%202021-09-24%20at%2017.46.51.png" style="border-radius: 10px; height: 60%;" />
  <p style="text-align: center; font-size: 0.6em; color: grey;">Asteria 2024</p>
</div>

<!--s-->

<div class = "header-slide">

# Rise and Fall of the OLAP Cube

</div>

<!--s-->

## OLAP Cube

An **OLAP Cube** is a multi-dimensional array of data used in business intelligence. Instead of storing data in a tabular format, cubes allow for complex calculations, trend analysis, and sophisticated data modeling. This is because OLAP cubes can store data in multiple dimensions, allowing users to analyze data from different perspectives without the need for complex joins.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

A common workflow to build OLAP cubes is:

1. **Extract**: Data is extracted from various sources, such as databases, spreadsheets, or flat files.
2. **Transform**: Data is cleaned, transformed, and (often) aggregated.
3. **Load**: Data is loaded into the OLAP cube.

</div>
<div class="c2" style = "width: 50%">

<div class = "col-wrapper col-centered">
<img style="border-radius: 10px;" src="https://i0.wp.com/olap.com/wp-content/uploads/2019/06/olap-3d-cube.png?fit=2016%2C1890&ssl=1" />
<p style="text-align: center; font-size: 0.6em; color: grey;">olap.com 2025</p>
</div>

</div>
</div>


<!--s-->

## OLAP Cube Definition

<div class = "col-wrapper">
<div class="c1" style="width: 50%; height: auto;">

We can define OLAP as a function of axes. Consider a data cube with dimensions $(X, Y, Z)$. An OLAP operation can be represented as:

$$ f : (X, Y, Z) \rightarrow \text{W} $$

Where the result (W) is a subset or aggregation of the data based on the specified dimensions and measures.

</div>

<div class="c2" style="width: 50%; height: auto;">

<img style="border-radius: 10px;" src="https://i0.wp.com/olap.com/wp-content/uploads/2019/06/olap-3d-cube.png?fit=2016%2C1890&ssl=1" />
<p style="text-align: center; font-size: 0.6em; color: grey;">olap.com 2025</p>

</div>
</div>

<!--s-->

## OLAP Cube Operations

OLAP Cubes enable various operations to analyze data.

| Operation | Description | Example |
|-----------|-------------|---------|
| Drill-up | Aggregates data along a dimension | Monthly sales to quarterly sales |
| Drill-down | Decomposes data into finer levels | Quarterly sales to monthly sales |
| Slice | Selects a single dimension | Sales for a specific product |
| Dice | Selects two or more dimensions | Sales for a specific product in a specific region |

<!--s-->

## OLAP Cube Operations | Drill-down

Drill-down decomposes data into finer levels. 

Drilling down in *Outdoor protective equipment* reveals specific data inside the category (*insect repellent, Sunblock, First Aid*)

<img src = "https://upload.wikimedia.org/wikipedia/commons/9/9b/OLAP_drill_up%26down_en.png" style="border-radius: 10px"/>
<p style="text-align: center; font-size: 0.6em; color: grey;">Wikipedia 2023</p>

<!--s-->

## OLAP Cube Operations | Drill-up

Drill-up aggregates data along a dimension. Drilling up in *Outdoor protective equipment* reveals the total sales for the entire category.

<img src = "https://upload.wikimedia.org/wikipedia/commons/9/9b/OLAP_drill_up%26down_en.png" style="border-radius: 10px"/>
<p style="text-align: center; font-size: 0.6em; color: grey;">Wikipedia 2023</p>

<!--s-->

## OLAP Cube Operations | Slice

Slice selects a single dimension. Here we just want to see *2004* data.

<img src = "https://upload.wikimedia.org/wikipedia/commons/a/a6/OLAP_slicing_en.png" style="border-radius: 10px"/>
<p style="text-align: center; font-size: 0.6em; color: grey;">Wikipedia 2023</p>

<!--s-->

## OLAP Cube Operations | Dice

Dice selects two or more dimensions. Here you can see diced to only read *Accessories* $\rightarrow$ *Golf equipment*.

<img src = "https://upload.wikimedia.org/wikipedia/commons/c/c7/OLAP_dicing_en.png" style="border-radius: 10px"/>

<p style="text-align: center; font-size: 0.6em; color: grey;">Wikipedia 2023</p>

<!--s-->

## OLAP Cubes | Downfall

In recent years, the use of OLAP cubes has declined due to:

- **Complexity**: Building and maintaining OLAP cubes can be complex and time-consuming.
- **Data Volume**: The explosion of data volume and variety makes it challenging to pre-aggregate data.
- **Real-time Analytics**: The need for real-time data access and analytics has led to the adoption of more flexible data architectures.

Still, many organizations continue to use OLAP cubes for specific use cases, especially in traditional business analytics environments.

<!--s-->

<div class = "header-slide">

# Columnar Databases

</div>

<!--s-->

## Columnar Databases

**Columnar Databases** store data tables primarily by column rather than row. This storage approach is ideal for OLAP scenarios as it dramatically speeds up the querying of large datasets.

<div class="col-wrapper col-centered">
<img src = "https://storage.googleapis.com/gweb-cloudblog-publish/images/BigQuery_Explained_storage_options_2.max-700x700.png" style="border-radius: 10px"/>
<p style="text-align: center; font-size: 0.6em; color: grey;">Thallum, 2020</p>
</div>

<!--s-->

## Why Column-Based Databases?

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

### Row-based databases

Row-based databases store data in rows, which is efficient for transactional workloads but can be inefficient for analytical queries that often require scanning large amounts of data across multiple rows.

### Column-based databases

Column-based databases provide faster data retrieval and more effective data compression than traditional row-oriented databases, especially suited for read-oriented tasks.

</div>
<div class="c2 col-centered" style = "width: 50%">

<img src = "https://storage.googleapis.com/gweb-cloudblog-publish/images/BigQuery_Explained_storage_options_2.max-700x700.png" style="border-radius: 10px"/>
<p style="text-align: center; font-size: 0.6em; color: grey;">Thallum, 2020</p>

</div>
</div>

<!--s-->

## Why Column-Based Databases?

### Advantages of Columnar Storage

1. **Faster Query Performance**: Only the necessary columns are read, reducing I/O operations.

2. **Better Compression**: Similar data types are stored together, allowing for more efficient compression algorithms.

3. **Improved Analytics**: Columnar storage is optimized for analytical queries, making it easier to perform aggregations and calculations.

4. **Scalability**: Columnar databases can handle large volumes of data and scale horizontally by adding more nodes to the cluster.

<!--s-->

## Advances that make Columnar DBs feasible

1. **Data Compression**: Columnar DBs use techniques such as Run Length Encoding (essentially storing the number of times a value is repeated) and Dictionary Encoding (storing a dictionary of unique values and their corresponding indices) to reduce storage space and improve query performance.

2. **Vectorized Execution**: Utilizes CPU vector registers to process multiple data elements with a single instruction, enhancing performance. CPU vector registers are small, fast storage locations within the CPU that can hold multiple data points simultaneously.

3. **SIMD (Single Instruction, Multiple Data)**: SIMD architectures allow the same operation to simultaneously occur on multiple data points, crucial for achieving high performance on columnar storage through parallel processing.

<!--s-->

## Cloud-based Columnar Data Warehouse Services

### AWS Redshift

Uses columnar storage, massively parallel processing, and optimized compression to enhance performance.

### GCP BigQuery

Serverless, highly scalable, and cost-effective multi-cloud data warehouse designed for business agility.

### SnowFlake

Provides a unique architecture with a separation of compute and storage layers, allowing for scalable and elastic performance.

<!--s-->

<div class="header-slide">

# Summary

</div>

<!--s-->

## Summary

**Online Analytical Processing (OLAP)** is a powerful approach to data analysis that enables users to interactively explore and analyze large datasets. 

**OLAP cubes**, while once the gold standard for OLAP, have seen a decline in popularity due to their complexity and the rise of more flexible data architectures. 

**Columnar databases** have emerged as a key technology for OLAP, providing faster query performance and better compression.

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Exit Poll
  ## On a scale of 1-5, how confident are you with the **OLAP** concepts?

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Exit Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->