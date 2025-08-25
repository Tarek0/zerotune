"""
Chart Generator for Publication Analysis

This module generates publication-ready convergence charts showing performance
across optimization iterations for different methods (ZeroTune, TPE, etc.).
"""

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import t
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime


class PublicationChartGenerator:
    """
    Generator for publication-ready convergence charts.
    
    Creates charts showing average AUC performance across HPO iterations,
    comparing different methods (ZeroTune, warmstart TPE, standard TPE, etc.).
    
    Default settings optimized for publication:
    - only_increasing=True: Shows cumulative maximum (monotonic performance)
    - show_confidence_intervals=True: Displays statistical uncertainty
    - remove_headers=True: Clean design without titles/axis labels
    """
    
    def __init__(self, 
                 confidence_level: float = 0.95,
                 decimal_places: int = 4,
                 figure_width: int = 800,
                 figure_height: int = 600):
        """
        Initialize the chart generator.
        
        Args:
            confidence_level: Confidence level for confidence intervals (default: 0.95)
            decimal_places: Number of decimal places for display (default: 4)
            figure_width: Width of generated figures in pixels (default: 800)
            figure_height: Height of generated figures in pixels (default: 600)
        """
        self.confidence_level = confidence_level
        self.decimal_places = decimal_places
        self.figure_width = figure_width
        self.figure_height = figure_height
    
    def generate_convergence_charts(
        self,
        convergence_results: Dict[str, Any],
        benchmark_data: pd.DataFrame,
        algorithm_name: str,
        output_dir: str,
        dataset_ids: Optional[List[int]] = None,
        show_confidence_intervals: bool = True,
        show_individual_traces: bool = False,
        only_increasing: bool = True,
        remove_headers: bool = True,
        show_axis_labels: bool = True,
        save_charts: bool = True,
        include_warmstart: bool = True,
        chart_version: str = "v2"
    ) -> Dict[str, str]:
        """
        Generate convergence charts for all datasets or specified dataset IDs.
        
        Args:
            convergence_results: Results from CheckpointAnalyzer.analyze_convergence()
            benchmark_data: Benchmark results DataFrame with baseline performance
            algorithm_name: Name of the algorithm for titles and filenames
            output_dir: Directory to save chart files
            dataset_ids: Optional list of dataset IDs to generate charts for (default: all)
            show_confidence_intervals: Whether to show confidence intervals (default: True)
            show_individual_traces: Whether to show individual seed traces (default: False)
            only_increasing: Whether to show only increasing performance (default: True)
            remove_headers: Whether to remove plot titles (default: True)
            show_axis_labels: Whether to show axis labels (default: True)
            save_charts: Whether to save charts to files (default: True)
            include_warmstart: Whether to include warmstart method in charts (default: True)
            chart_version: Version identifier for filename ("v1" or "v2")
            
        Returns:
            Dictionary mapping dataset_id to chart file path
        """
        print("📊 Generating convergence charts...")
        
        if 'trial_data' not in convergence_results:
            print("❌ No trial data available for chart generation")
            return {}
        
        trial_data = convergence_results['trial_data']
        
        # Get available dataset IDs
        available_datasets = sorted(trial_data['dataset_id'].unique())
        
        if dataset_ids is None:
            dataset_ids = available_datasets
        else:
            # Filter to only available datasets
            dataset_ids = [did for did in dataset_ids if did in available_datasets]
        
        if not dataset_ids:
            print("❌ No valid dataset IDs found for chart generation")
            return {}
        
        print(f"📈 Generating charts for {len(dataset_ids)} datasets: {dataset_ids}")
        
        chart_files = {}
        
        for dataset_id in dataset_ids:
            print(f"   Creating chart for dataset {dataset_id}...")
            
            try:
                chart_path = self._generate_single_dataset_chart(
                    trial_data=trial_data,
                    benchmark_data=benchmark_data,
                    dataset_id=dataset_id,
                    algorithm_name=algorithm_name,
                    output_dir=output_dir,
                    show_confidence_intervals=show_confidence_intervals,
                    show_individual_traces=show_individual_traces,
                    only_increasing=only_increasing,
                    remove_headers=remove_headers,
                    show_axis_labels=show_axis_labels,
                    save_chart=save_charts,
                    include_warmstart=include_warmstart,
                    chart_version=chart_version
                )
                
                if chart_path:
                    chart_files[dataset_id] = chart_path
                    print(f"   ✅ Chart saved: {os.path.basename(chart_path)}")
                else:
                    print(f"   ⚠️  Could not generate chart for dataset {dataset_id}")
                    
            except Exception as e:
                print(f"   ❌ Error generating chart for dataset {dataset_id}: {str(e)}")
        
        print(f"✅ Generated {len(chart_files)} convergence charts")
        return chart_files
    
    def _generate_single_dataset_chart(
        self,
        trial_data: pd.DataFrame,
        benchmark_data: pd.DataFrame,
        dataset_id: int,
        algorithm_name: str,
        output_dir: str,
        show_confidence_intervals: bool = True,
        show_individual_traces: bool = False,
        only_increasing: bool = True,
        remove_headers: bool = True,
        show_axis_labels: bool = True,
        save_chart: bool = True,
        include_warmstart: bool = True,
        chart_version: str = "v2"
    ) -> Optional[str]:
        """
        Generate a convergence chart for a single dataset.
        
        Args:
            trial_data: DataFrame with trial data from convergence analysis
            benchmark_data: Benchmark results DataFrame
            dataset_id: Dataset ID to generate chart for
            algorithm_name: Algorithm name for titles and filenames
            output_dir: Output directory for saving charts
            show_confidence_intervals: Whether to show confidence intervals
            show_individual_traces: Whether to show individual seed traces
            only_increasing: Whether to show only increasing performance
            remove_headers: Whether to remove plot titles
            show_axis_labels: Whether to show axis labels
            save_chart: Whether to save the chart to file
            include_warmstart: Whether to include warmstart method in charts
            chart_version: Version identifier for filename ("v1" or "v2")
            
        Returns:
            Path to saved chart file or None if generation failed
        """
        # Filter data for this dataset
        dataset_trials = trial_data[trial_data['dataset_id'] == dataset_id].copy()
        dataset_benchmark = benchmark_data[benchmark_data['dataset_id'] == dataset_id].copy()
        
        if len(dataset_trials) == 0:
            print(f"   No trial data found for dataset {dataset_id}")
            return None
        
        if len(dataset_benchmark) == 0:
            print(f"   No benchmark data found for dataset {dataset_id}")
            return None
        
        # Get dataset name if available
        dataset_name = dataset_benchmark['dataset_name'].iloc[0] if 'dataset_name' in dataset_benchmark.columns else f"Dataset {dataset_id}"
        
        # Create figure
        fig = go.Figure()
        
        # Process trial data by method
        all_methods = dataset_trials['method'].unique()
        
        # Filter methods based on version
        if include_warmstart:
            methods = all_methods  # Include both warmstart and standard
        else:
            methods = [m for m in all_methods if m != 'warmstart']  # Only standard TPE
        
        method_colors = {
            'warmstart': '#ff7f0e',  # Orange for warmstart (ZT2+TPE)
            'standard': '#1f77b4',   # Blue for standard TPE
        }
        method_names = {
            'warmstart': 'ZT2 + TPE',
            'standard': 'TPE'
        }
        
        for method in methods:
            method_trials = dataset_trials[dataset_trials['method'] == method].copy()
            
            if len(method_trials) == 0:
                continue
            
            color = method_colors.get(method, '#2ca02c')  # Default green
            display_name = method_names.get(method, method.title())
            
            # Process by seed
            seeds = sorted(method_trials['seed'].unique())
            
            # Plot individual traces if requested
            if show_individual_traces:
                for seed in seeds:
                    seed_trials = method_trials[method_trials['seed'] == seed].copy()
                    seed_trials = seed_trials.sort_values('number')
                    
                    # Calculate cumulative maximum
                    seed_trials['cummax_value'] = seed_trials['value'].cummax()
                    
                    fig.add_trace(go.Scatter(
                        x=seed_trials['number'],
                        y=seed_trials['cummax_value'],
                        mode='lines',
                        name=f'{display_name} Seed {seed}',
                        line=dict(color=color, width=1, dash='dot'),
                        opacity=0.3,
                        showlegend=False
                    ))
            
            # Calculate mean performance across seeds
            stats_data = self._calculate_method_statistics(method_trials, only_increasing)
            
            if len(stats_data) == 0:
                continue
            
            # Add confidence intervals if requested
            if show_confidence_intervals and len(stats_data) > 0:
                self._add_confidence_interval(fig, stats_data, color, display_name)
            
            # Add main performance line
            fig.add_trace(go.Scatter(
                x=stats_data['trial_number'],
                y=stats_data['mean_cummax'],
                mode='lines+markers',
                name=display_name,
                line=dict(color=color, width=2),
                marker=dict(size=8)
            ))
        
        # Add baseline performance lines from benchmark data
        self._add_baseline_performance_lines(fig, dataset_benchmark, dataset_trials)
        
        # Update layout
        fig.update_layout(
            title=None if remove_headers else f'{algorithm_name} Performance Convergence<br>Dataset: {dataset_name}',
            xaxis_title='Iteration Number' if show_axis_labels else None,
            yaxis_title='Performance (AUC)' if show_axis_labels else None,
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white',
            width=self.figure_width,
            height=self.figure_height,
            legend=dict(
                x=1,
                y=0,
                xanchor='right',
                yanchor='bottom'
            )
        )
        
        # Style axes
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        
        # Save chart if requested
        if save_chart:
            # Create plots subdirectory
            plots_dir = os.path.join(output_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            
            # Use specific naming pattern based on version
            if chart_version == "v1":
                filename = f"ZT2_vs_TPE_Performance_Dataset_{dataset_id}.pdf"
            else:  # v2
                filename = f"ZT2_WS_vs_TPE_Performance_Dataset_{dataset_id}.pdf"
            
            filepath = os.path.join(plots_dir, filename)
            
            fig.write_image(filepath)
            return filepath
        
        return None
    
    def _calculate_method_statistics(self, method_trials: pd.DataFrame, only_increasing: bool = True) -> pd.DataFrame:
        """
        Calculate statistics for a method across seeds and trial numbers.
        
        Args:
            method_trials: DataFrame with trial data for a single method
            
        Returns:
            DataFrame with statistics by trial number
        """
        # Sort by seed and trial number
        method_trials = method_trials.sort_values(['seed', 'number'])
        
        # Calculate cumulative maximum for each seed (this gives us the best performance achieved so far)
        method_trials['cummax_value'] = method_trials.groupby('seed')['value'].cummax()
        
        # Get the maximum trial number, limited to 20 for smooth curves
        max_trials = min(20, method_trials['number'].max())
        
        # Generate statistics for each iteration from 0 to max_trials (include iteration 0)
        stats_list = []
        
        # Pre-calculate cummax per seed for statistically correct CI calculation
        seed_cummax_data = {}
        for seed in method_trials['seed'].unique():
            seed_data = method_trials[method_trials['seed'] == seed].copy()
            seed_data = seed_data.sort_values('number')
            seed_data['cummax_value'] = seed_data['value'].cummax()
            seed_cummax_data[seed] = seed_data
        
        for trial_num in range(0, max_trials + 1):
            # Get the cummax performance for each seed at this trial
            seed_cummax_performances = []
            
            for seed in method_trials['seed'].unique():
                seed_data = seed_cummax_data[seed]
                # Get trials up to this trial number for this seed
                trials_up_to_now = seed_data[seed_data['number'] <= trial_num]
                
                if len(trials_up_to_now) > 0:
                    # Get the cummax performance at this trial
                    cummax_performance = trials_up_to_now['cummax_value'].iloc[-1]  # Last value is the cummax up to this trial
                    seed_cummax_performances.append(cummax_performance)
            
            if len(seed_cummax_performances) > 0:
                # Calculate statistics directly on cummax values (statistically correct)
                mean_cummax = np.mean(seed_cummax_performances)
                std_cummax = np.std(seed_cummax_performances, ddof=1)  # Sample standard deviation
                count = len(seed_cummax_performances)
                sem_cummax = std_cummax / np.sqrt(count)
                
                # Calculate confidence intervals directly on cummax values
                degrees_freedom = count - 1
                t_critical = t.ppf((1 + self.confidence_level) / 2, degrees_freedom)
                ci_cummax = t_critical * sem_cummax
                
                stats_list.append({
                    'trial_number': trial_num,
                    'mean_cummax': mean_cummax,
                    'std_cummax': std_cummax,
                    'sem_cummax': sem_cummax,
                    'ci': ci_cummax,
                    'count': count
                })
        
        stats = pd.DataFrame(stats_list)
        
        # For non-increasing mode, we would need to calculate regular means
        # But since we primarily use only_increasing=True, this approach is optimal
        if not only_increasing:
            # Fallback: calculate regular means for non-cummax case
            # This is rarely used, so we'll keep it simple
            stats['mean'] = stats['mean_cummax']  # Use cummax as approximation
        
        # Rename for clarity
        stats = stats.rename(columns={'number': 'trial_number'})
        
        return stats
    
    def _add_confidence_interval(self, fig: go.Figure, stats_data: pd.DataFrame, color: str, method_name: str):
        """
        Add confidence interval shading to the figure.
        
        Args:
            fig: Plotly figure to add confidence interval to
            stats_data: DataFrame with statistics including confidence intervals
            color: Base color for the confidence interval
            method_name: Method name for hover text
        """
        if len(stats_data) == 0:
            return
        
        # Convert color to RGBA for transparency
        if color.startswith('#'):
            # Convert hex to RGB
            hex_color = color.lstrip('#')
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            rgba_color = f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.2)'  # Standard opacity for confidence intervals
        else:
            rgba_color = f'rgba(128, 128, 128, 0.2)'  # Fallback with standard opacity
        
        # Create confidence interval
        x_values = pd.concat([stats_data['trial_number'], stats_data['trial_number'][::-1]])
        y_upper = stats_data['mean_cummax'] + stats_data['ci']
        y_lower = stats_data['mean_cummax'] - stats_data['ci']
        y_values = pd.concat([y_upper, y_lower[::-1]])
        
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            fill='toself',
            fillcolor=rgba_color,
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False,
            name=f'{method_name} CI'
        ))
    
    def _add_baseline_performance_lines(self, fig: go.Figure, dataset_benchmark: pd.DataFrame, dataset_trials: pd.DataFrame):
        """
        Add horizontal baseline performance lines for ZeroTune, Random, etc.
        
        Args:
            fig: Plotly figure to add baseline lines to
            dataset_benchmark: Benchmark data for this dataset
            dataset_trials: Trial data for this dataset (to get TPE iteration 0 performance)
        """
        if len(dataset_benchmark) == 0:
            return
        
        # Get trial number range for horizontal lines from actual data
        if len(fig.data) > 0:
            # Get x-range from existing traces
            all_x_values = []
            for trace in fig.data:
                if hasattr(trace, 'x') and trace.x is not None:
                    all_x_values.extend(trace.x)
            if all_x_values:
                x_range = [min(all_x_values), max(all_x_values)]
            else:
                x_range = [1, 20]  # Default fallback
        else:
            x_range = [1, 20]  # Default fallback
        
        # Add ZeroTune baseline (ZT2) - use benchmark data which should now match trial 0
        zt_performance = None
        if 'auc_predicted' in dataset_benchmark.columns:
            zt_performance = dataset_benchmark['auc_predicted'].iloc[0]
        
        if zt_performance is not None and not pd.isna(zt_performance):
            fig.add_trace(go.Scatter(
                x=x_range,
                y=[zt_performance, zt_performance],
                mode='lines',
                name='ZT2',
                line=dict(color='red', width=2, dash='dash'),
                showlegend=True
            ))
        
        # Add Random baseline - use TPE's iteration 0 performance (TPE's first iteration is effectively random)
        tpe_iteration_0_performance = None
        if len(dataset_trials) > 0:
            # Get standard TPE trials at iteration 0
            standard_trials = dataset_trials[dataset_trials['method'] == 'standard']
            if len(standard_trials) > 0:
                iteration_0_trials = standard_trials[standard_trials['number'] == 0]
                if len(iteration_0_trials) > 0:
                    tpe_iteration_0_performance = iteration_0_trials['value'].mean()
        
        # Fallback to benchmark random if TPE iteration 0 not available
        if tpe_iteration_0_performance is None and 'auc_random' in dataset_benchmark.columns:
            tpe_iteration_0_performance = dataset_benchmark['auc_random'].iloc[0]
        
        if tpe_iteration_0_performance is not None and not pd.isna(tpe_iteration_0_performance):
            fig.add_trace(go.Scatter(
                x=[0, x_range[1]],  # Start at iteration 0
                y=[tpe_iteration_0_performance, tpe_iteration_0_performance],
                mode='lines',
                name='Random',
                line=dict(color='green', width=2, dash='dash'),
                showlegend=True
            ))
        
        # Add Default baseline if available
        if 'auc_default' in dataset_benchmark.columns:
            default_performance = dataset_benchmark['auc_default'].iloc[0]
            if not pd.isna(default_performance):
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=[default_performance, default_performance],
                    mode='lines',
                    name='Default',
                    line=dict(color='orange', width=2, dash='dash'),
                    showlegend=True
                ))
    
    def generate_aggregated_convergence_chart(
        self,
        convergence_results: Dict[str, Any],
        benchmark_data: pd.DataFrame,
        algorithm_name: str,
        output_dir: str,
        save_chart: bool = True
    ) -> Optional[str]:
        """
        Generate an aggregated convergence chart showing average performance across all datasets.
        
        Args:
            convergence_results: Results from CheckpointAnalyzer.analyze_convergence()
            benchmark_data: Benchmark results DataFrame with baseline performance
            algorithm_name: Algorithm name for titles and filenames
            output_dir: Directory to save chart files
            save_chart: Whether to save the chart to file
            
        Returns:
            Path to saved chart file or None if generation failed
        """
        print("📊 Generating aggregated convergence chart...")
        
        if 'trial_data' not in convergence_results:
            print("❌ No trial data available for aggregated chart")
            return None
        
        trial_data = convergence_results['trial_data']
        
        if len(trial_data) == 0:
            print("❌ Trial data is empty")
            return None
        
        # Create figure
        fig = go.Figure()
        
        # Get available methods from trial data
        methods = trial_data['method'].unique()
        method_colors = {
            'warmstart': '#ff7f0e',  # Orange
            'standard': '#1f77b4',   # Blue
        }
        method_names = {
            'warmstart': 'ZT2 + TPE',
            'standard': 'TPE'
        }
        
        for method in methods:
            method_trials = trial_data[trial_data['method'] == method].copy()
            
            if len(method_trials) == 0:
                continue
            
            color = method_colors.get(method, '#2ca02c')
            display_name = method_names.get(method, method.title())
            
            # Calculate statistics for this method across all datasets
            stats_data = self._calculate_method_statistics(method_trials, only_increasing=True)
            
            if len(stats_data) == 0:
                continue
            
            # Add confidence intervals
            self._add_confidence_interval(fig, stats_data, color, display_name)
            
            # Add main line
            fig.add_trace(go.Scatter(
                x=stats_data['trial_number'],
                y=stats_data['mean_cummax'],
                mode='lines+markers',
                name=display_name,
                line=dict(color=color, width=2),
                marker=dict(size=8)
            ))
        
        # Add ZT2 baseline (zero-shot performance) as horizontal line
        if benchmark_data is not None and 'auc_predicted' in benchmark_data.columns:
            zt_performance = benchmark_data['auc_predicted'].mean()  # Average across all datasets
            if not pd.isna(zt_performance):
                # Get x-range from the trial data (1 to 20)
                x_range = [1, 20]
                
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=[zt_performance, zt_performance],
                    mode='lines',
                    name='ZT2',
                    line=dict(color='red', width=2, dash='dash'),
                    showlegend=True
                ))
        
        # Update layout
        fig.update_layout(
            title=None,  # Remove headers by default
            xaxis_title=None,  # Remove headers by default
            yaxis_title=None,  # Remove headers by default
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white',
            width=self.figure_width,
            height=self.figure_height,
            legend=dict(
                x=1,
                y=0,
                xanchor='right',
                yanchor='bottom'
            )
        )
        
        # Style axes
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        
        # Save chart if requested
        if save_chart:
            # Create plots subdirectory
            plots_dir = os.path.join(output_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{algorithm_name}_all_datasets_aggregated_convergence_{timestamp}.pdf"
            filepath = os.path.join(plots_dir, filename)
            
            fig.write_image(filepath)
            return filepath
        
        return None
    
    def _hex_to_rgba(self, hex_color: str, alpha: float) -> str:
        """
        Convert hex color to RGBA string.
        
        Args:
            hex_color: Hex color string (e.g., '#1f77b4')
            alpha: Alpha value for transparency (0.0 to 1.0)
            
        Returns:
            RGBA color string
        """
        if not hex_color.startswith('#'):
            return f'rgba(128, 128, 128, {alpha})'  # Fallback
        
        try:
            hex_color = hex_color.lstrip('#')
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            return f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})'
        except:
            return f'rgba(128, 128, 128, {alpha})'  # Fallback
    
    def generate_comparison_chart(
        self,
        convergence_results: Dict[str, Any],
        benchmark_data: pd.DataFrame,
        algorithm_name: str,
        output_dir: str,
        comparison_type: str = "zt_vs_tpe",
        save_chart: bool = True
    ) -> Optional[str]:
        """
        Generate a comparison chart showing win rates across checkpoints.
        
        Args:
            convergence_results: Results from convergence analysis
            benchmark_data: Benchmark results DataFrame
            algorithm_name: Algorithm name for titles and filenames
            output_dir: Directory to save chart files
            comparison_type: Type of comparison ('zt_vs_tpe' or 'aggregated_comparison')
            save_chart: Whether to save the chart to file
            
        Returns:
            Path to saved chart file or None if generation failed
        """
        print(f"📊 Generating {comparison_type} comparison chart...")
        
        if 'convergence_tables' not in convergence_results:
            print("❌ No convergence tables available")
            return None
        
        tables = convergence_results['convergence_tables']
        
        if comparison_type not in tables:
            print(f"❌ No {comparison_type} table available")
            return None
        
        comparison_table = tables[comparison_type]
        
        if len(comparison_table) == 0:
            print(f"❌ {comparison_type} table is empty")
            return None
        
        # Create figure
        fig = go.Figure()
        
        # Instead of plotting win percentages, plot actual average AUC performance
        # Get the method comparison table which has average AUC values
        method_comparison_table = tables.get('method_comparison', pd.DataFrame())
        
        if len(method_comparison_table) > 0:
            # Add warmstart (ZT+TPE) performance line
            if 'warmstart_mean' in method_comparison_table.columns:
                valid_data = method_comparison_table.dropna(subset=['warmstart_mean'])
                
                fig.add_trace(go.Scatter(
                    x=valid_data['checkpoint'],
                    y=valid_data['warmstart_mean'],
                    mode='lines+markers',
                    name='ZT+TPE',
                    line=dict(color='orange', width=2),
                    marker=dict(size=8)
                ))
            
            # Add standard TPE performance line
            if 'standard_mean' in method_comparison_table.columns:
                valid_data = method_comparison_table.dropna(subset=['standard_mean'])
                
                fig.add_trace(go.Scatter(
                    x=valid_data['checkpoint'],
                    y=valid_data['standard_mean'],
                    mode='lines+markers',
                    name='TPE',
                    line=dict(color='blue', width=2),
                    marker=dict(size=8)
                ))
        
        # Add ZT2 baseline (zero-shot performance) as horizontal line
        if benchmark_data is not None and 'auc_predicted' in benchmark_data.columns:
            zt_performance = benchmark_data['auc_predicted'].mean()  # Average across all datasets
            if not pd.isna(zt_performance):
                # Get x-range from the method comparison data
                if len(method_comparison_table) > 0:
                    x_range = [method_comparison_table['checkpoint'].min(), method_comparison_table['checkpoint'].max()]
                else:
                    x_range = [1, 20]  # Default fallback
                
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=[zt_performance, zt_performance],
                    mode='lines',
                    name='ZT2',
                    line=dict(color='red', width=2, dash='dash'),
                    showlegend=True
                ))
        
        # Update layout
        title_map = {
            'zt_vs_tpe': 'ZT2 vs TPE Performance Comparison',
            'aggregated_comparison': 'ZT+TPE vs TPE Performance Comparison'
        }
        
        fig.update_layout(
            title=None,  # Remove headers by default
            xaxis_title=None,  # Remove headers by default
            yaxis_title=None,  # Remove headers by default
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white',
            width=self.figure_width,
            height=self.figure_height,
            legend=dict(
                x=1,
                y=0,
                xanchor='right',
                yanchor='bottom'
            )
        )
        
        # Style axes
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        
        # Save chart if requested
        if save_chart:
            # Create plots subdirectory
            plots_dir = os.path.join(output_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{algorithm_name}_all_datasets_{comparison_type}_comparison_{timestamp}.pdf"
            filepath = os.path.join(plots_dir, filename)
            
            fig.write_image(filepath)
            return filepath
        
        return None
    
    def generate_all_charts(
        self,
        convergence_results: Dict[str, Any],
        benchmark_data: pd.DataFrame,
        algorithm_name: str,
        output_dir: str,
        dataset_ids: Optional[List[int]] = None,
        include_individual_datasets: bool = True,
        include_aggregated: bool = False,
        include_comparison_charts: bool = False,
        generate_both_versions: bool = True,
        **chart_options
    ) -> Dict[str, str]:
        """
        Generate all publication charts.
        
        Args:
            convergence_results: Results from convergence analysis
            benchmark_data: Benchmark results DataFrame
            algorithm_name: Algorithm name for titles and filenames
            output_dir: Directory to save chart files
            dataset_ids: Optional list of dataset IDs (default: all available)
            include_individual_datasets: Whether to generate individual dataset charts
            include_aggregated: Whether to generate aggregated chart (default: False)
            include_comparison_charts: Whether to generate comparison charts (default: False)
            generate_both_versions: Whether to generate both chart versions (default: True)
            **chart_options: Additional options passed to chart generation methods
            
        Returns:
            Dictionary mapping chart type to file paths
        """
        print("📊 Generating all publication charts...")
        
        all_chart_files = {}
        
        # Generate individual dataset charts
        if include_individual_datasets:
            if generate_both_versions:
                # Version 1: ZT2 vs TPE (without warmstart)
                dataset_charts_v1 = self.generate_convergence_charts(
                    convergence_results=convergence_results,
                    benchmark_data=benchmark_data,
                    algorithm_name=algorithm_name,
                    output_dir=output_dir,
                    dataset_ids=dataset_ids,
                    include_warmstart=False,
                    chart_version="v1",
                    **chart_options
                )
                
                # Version 2: ZT2 + TPE vs TPE (with warmstart)
                dataset_charts_v2 = self.generate_convergence_charts(
                    convergence_results=convergence_results,
                    benchmark_data=benchmark_data,
                    algorithm_name=algorithm_name,
                    output_dir=output_dir,
                    dataset_ids=dataset_ids,
                    include_warmstart=True,
                    chart_version="v2",
                    **chart_options
                )
                
                # Combine both versions
                for dataset_id, filepath in dataset_charts_v1.items():
                    all_chart_files[f'dataset_{dataset_id}_v1'] = filepath
                for dataset_id, filepath in dataset_charts_v2.items():
                    all_chart_files[f'dataset_{dataset_id}_v2'] = filepath
            else:
                # Original single version
                dataset_charts = self.generate_convergence_charts(
                    convergence_results=convergence_results,
                    benchmark_data=benchmark_data,
                    algorithm_name=algorithm_name,
                    output_dir=output_dir,
                    dataset_ids=dataset_ids,
                    **chart_options
                )
                
                for dataset_id, filepath in dataset_charts.items():
                    all_chart_files[f'dataset_{dataset_id}'] = filepath
        
        # Generate aggregated chart
        if include_aggregated:
            # Filter out unsupported options for aggregated chart
            aggregated_options = {k: v for k, v in chart_options.items() if k in ['save_chart']}
            aggregated_chart = self.generate_aggregated_convergence_chart(
                convergence_results=convergence_results,
                benchmark_data=benchmark_data,
                algorithm_name=algorithm_name,
                output_dir=output_dir,
                **aggregated_options
            )
            
            if aggregated_chart:
                all_chart_files['aggregated_convergence'] = aggregated_chart
        
        # Generate comparison charts
        if include_comparison_charts:
            for comparison_type in ['zt_vs_tpe_comparison', 'aggregated_comparison']:
                # Filter out unsupported options for comparison chart
                comparison_options = {k: v for k, v in chart_options.items() if k in ['save_chart']}
                comparison_chart = self.generate_comparison_chart(
                    convergence_results=convergence_results,
                    benchmark_data=benchmark_data,
                    algorithm_name=algorithm_name,
                    output_dir=output_dir,
                    comparison_type=comparison_type,
                    **comparison_options
                )
                
                if comparison_chart:
                    all_chart_files[comparison_type] = comparison_chart
        
        print(f"✅ Generated {len(all_chart_files)} publication charts")
        return all_chart_files 