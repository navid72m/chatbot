// src/components/data/Chart.jsx

import React, { useEffect, useState, useRef } from 'react';
import PropTypes from 'prop-types';
import { 
  LineChart, 
  Line, 
  BarChart,
  Bar,
  AreaChart,
  Area,
  PieChart,
  Pie,
  Cell,
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend,
  ResponsiveContainer,
  Label
} from 'recharts';
import Card from '../common/Card';

// Default color palette
const DEFAULT_COLORS = [
  '#4f46e5', // primary
  '#10b981', // success
  '#f59e0b', // warning
  '#ef4444', // danger
  '#3b82f6', // info
  '#8b5cf6', // purple
  '#ec4899', // pink
  '#14b8a6', // teal
  '#f97316', // orange
  '#6366f1', // indigo
];

const Chart = ({
  type = 'line',
  data = [],
  series = [],
  xKey = 'name',
  yUnit = '',
  height = 300,
  colors = DEFAULT_COLORS,
  title,
  subtitle,
  showLegend = true,
  showGrid = true,
  showTooltip = true,
  stacked = false,
  percentage = false,
  formatValue,
  formatTooltip,
  className = '',
  onDataPointClick,
  emptyStateMessage = 'No data available',
  loading = false,
  error = null,
}) => {
  const [chartData, setChartData] = useState([]);
  const chartRef = useRef(null);
  
  // Process data and series configuration
  useEffect(() => {
    if (data && data.length > 0) {
      setChartData(data);
    } else {
      setChartData([]);
    }
  }, [data]);
  
  // Format value for axis and tooltips
  const formatChartValue = (value) => {
    if (formatValue) {
      return formatValue(value);
    }
    
    if (percentage) {
      return `${value}%`;
    }
    
    if (yUnit) {
      return `${value}${yUnit}`;
    }
    
    return value;
  };
  
  // Custom tooltip formatter
  const renderTooltipContent = (props) => {
    const { payload, label, active } = props;
    
    if (!active || !payload || payload.length === 0) {
      return null;
    }
    
    if (formatTooltip) {
      return formatTooltip(props);
    }
    
    return (
      <div className="chart-tooltip">
        <p className="chart-tooltip-label">{label}</p>
        {payload.map((entry, index) => (
          <div key={`tooltip-${index}`} className="chart-tooltip-item">
            <span
              className="chart-tooltip-bullet"
              style={{ backgroundColor: entry.color }}
            />
            <span className="chart-tooltip-name">{entry.name}:</span>
            <span className="chart-tooltip-value">
              {formatChartValue(entry.value)}
            </span>
          </div>
        ))}
      </div>
    );
  };
  
  // Render loading state
  if (loading) {
    return (
      <Card
        title={title}
        subtitle={subtitle}
        className={`chart-card ${className}`}
      >
        <div className="chart-loading">
          <div className="chart-loading-spinner"></div>
          <p>Loading chart data...</p>
        </div>
      </Card>
    );
  }
  
  // Render error state
  if (error) {
    return (
      <Card
        title={title}
        subtitle={subtitle}
        className={`chart-card ${className}`}
      >
        <div className="chart-error">
          <p>Error loading chart data:</p>
          <p>{error}</p>
        </div>
      </Card>
    );
  }
  
  // Render empty state
  if (!chartData || chartData.length === 0) {
    return (
      <Card
        title={title}
        subtitle={subtitle}
        className={`chart-card ${className}`}
      >
        <div className="chart-empty-state">
          <p>{emptyStateMessage}</p>
        </div>
      </Card>
    );
  }
  
  // Render chart based on type
  const renderChart = () => {
    switch (type) {
      case 'bar':
        return (
          <ResponsiveContainer width="100%" height={height}>
            <BarChart
              data={chartData}
              margin={{ top: 10, right: 30, left: 0, bottom: 20 }}
              onClick={onDataPointClick ? (data) => onDataPointClick(data) : undefined}
              ref={chartRef}
            >
              {showGrid && <CartesianGrid strokeDasharray="3 3" vertical={false} />}
              
              <XAxis
                dataKey={xKey}
                tick={{ fontSize: 12 }}
                tickLine={false}
                axisLine={{ stroke: '#E5E7EB' }}
              />
              
              <YAxis
                tick={{ fontSize: 12 }}
                tickLine={false}
                axisLine={{ stroke: '#E5E7EB' }}
                tickFormatter={formatChartValue}
              >
                {yUnit && <Label value={yUnit} angle={-90} position="insideLeft" style={{ textAnchor: 'middle' }} />}
              </YAxis>
              
              {showTooltip && <Tooltip content={renderTooltipContent} />}
              {showLegend && <Legend wrapperStyle={{ fontSize: 12, paddingTop: 20 }} />}
              
              {series.map((item, index) => (
                <Bar
                  key={item.dataKey}
                  dataKey={item.dataKey}
                  name={item.name || item.dataKey}
                  fill={item.color || colors[index % colors.length]}
                  stackId={stacked ? 'stack' : undefined}
                  radius={[4, 4, 0, 0]}
                  barSize={stacked ? 30 : 20}
                />
              ))}
            </BarChart>
          </ResponsiveContainer>
        );
      
      case 'area':
        return (
          <ResponsiveContainer width="100%" height={height}>
            <AreaChart
              data={chartData}
              margin={{ top: 10, right: 30, left: 0, bottom: 20 }}
              onClick={onDataPointClick ? (data) => onDataPointClick(data) : undefined}
              ref={chartRef}
            >
              {showGrid && <CartesianGrid strokeDasharray="3 3" vertical={false} />}
              
              <XAxis
                dataKey={xKey}
                tick={{ fontSize: 12 }}
                tickLine={false}
                axisLine={{ stroke: '#E5E7EB' }}
              />
              
              <YAxis
                tick={{ fontSize: 12 }}
                tickLine={false}
                axisLine={{ stroke: '#E5E7EB' }}
                tickFormatter={formatChartValue}
              >
                {yUnit && <Label value={yUnit} angle={-90} position="insideLeft" style={{ textAnchor: 'middle' }} />}
              </YAxis>
              
              {showTooltip && <Tooltip content={renderTooltipContent} />}
              {showLegend && <Legend wrapperStyle={{ fontSize: 12, paddingTop: 20 }} />}
              
              {series.map((item, index) => (
                <Area
                  key={item.dataKey}
                  type="monotone"
                  dataKey={item.dataKey}
                  name={item.name || item.dataKey}
                  fill={item.color || colors[index % colors.length]}
                  stroke={item.color || colors[index % colors.length]}
                  fillOpacity={0.2}
                  stackId={stacked ? 'stack' : undefined}
                />
              ))}
            </AreaChart>
          </ResponsiveContainer>
        );
      
      case 'pie':
        return (
          <ResponsiveContainer width="100%" height={height}>
            <PieChart
              margin={{ top: 10, right: 30, left: 0, bottom: 20 }}
              onClick={onDataPointClick ? (data) => onDataPointClick(data) : undefined}
              ref={chartRef}
            >
              <Pie
                data={chartData}
                nameKey={xKey}
                dataKey={series[0]?.dataKey || 'value'}
                cx="50%"
                cy="50%"
                outerRadius={80}
                innerRadius={percentage ? 60 : 0}
                label={({ name, percent }) => 
                  percentage ? `${name}: ${(percent * 100).toFixed(0)}%` : name
                }
                labelLine={!percentage}
              >
                {chartData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={colors[index % colors.length]}
                  />
                ))}
              </Pie>
              
              {showTooltip && <Tooltip content={renderTooltipContent} />}
              {showLegend && <Legend wrapperStyle={{ fontSize: 12, paddingTop: 20 }} />}
            </PieChart>
          </ResponsiveContainer>
        );
      
      case 'line':
      default:
        return (
          <ResponsiveContainer width="100%" height={height}>
            <LineChart
              data={chartData}
              margin={{ top: 10, right: 30, left: 0, bottom: 20 }}
              onClick={onDataPointClick ? (data) => onDataPointClick(data) : undefined}
              ref={chartRef}
            >
              {showGrid && <CartesianGrid strokeDasharray="3 3" vertical={false} />}
              
              <XAxis
                dataKey={xKey}
                tick={{ fontSize: 12 }}
                tickLine={false}
                axisLine={{ stroke: '#E5E7EB' }}
              />
              
              <YAxis
                tick={{ fontSize: 12 }}
                tickLine={false}
                axisLine={{ stroke: '#E5E7EB' }}
                tickFormatter={formatChartValue}
              >
                {yUnit && <Label value={yUnit} angle={-90} position="insideLeft" style={{ textAnchor: 'middle' }} />}
              </YAxis>
              
              {showTooltip && <Tooltip content={renderTooltipContent} />}
              {showLegend && <Legend wrapperStyle={{ fontSize: 12, paddingTop: 20 }} />}
              
              {series.map((item, index) => (
                <Line
                  key={item.dataKey}
                  type="monotone"
                  dataKey={item.dataKey}
                  name={item.name || item.dataKey}
                  stroke={item.color || colors[index % colors.length]}
                  activeDot={{ r: 8 }}
                  dot={{ r: 4 }}
                  strokeWidth={2}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        );
    }
  };
  
  return (
    <Card
      title={title}
      subtitle={subtitle}
      className={`chart-card ${className}`}
    >
      <div className="chart-container">{renderChart()}</div>
    </Card>
  );
};

Chart.propTypes = {
  type: PropTypes.oneOf(['line', 'bar', 'area', 'pie']),
  data: PropTypes.array.isRequired,
  series: PropTypes.arrayOf(
    PropTypes.shape({
      dataKey: PropTypes.string.isRequired,
      name: PropTypes.string,
      color: PropTypes.string,
    })
  ).isRequired,
  xKey: PropTypes.string,
  yUnit: PropTypes.string,
  height: PropTypes.number,
  colors: PropTypes.array,
  title: PropTypes.node,
  subtitle: PropTypes.node,
  showLegend: PropTypes.bool,
  showGrid: PropTypes.bool,
  showTooltip: PropTypes.bool,
  stacked: PropTypes.bool,
  percentage: PropTypes.bool,
  formatValue: PropTypes.func,
  formatTooltip: PropTypes.func,
  className: PropTypes.string,
  onDataPointClick: PropTypes.func,
  emptyStateMessage: PropTypes.string,
  loading: PropTypes.bool,
  error: PropTypes.string,
};

export default Chart;