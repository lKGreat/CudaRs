using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;

namespace CudaRS.Paddle;

/// <summary>
/// Parses PaddlePaddle inference.yml configuration files
/// </summary>
public sealed class PaddleYamlParser
{
    /// <summary>
    /// Parse YAML configuration from file
    /// </summary>
    public static Dictionary<string, object> ParseFile(string yamlFilePath)
    {
        if (!File.Exists(yamlFilePath))
            throw new FileNotFoundException($"YAML file not found: {yamlFilePath}");

        var content = File.ReadAllText(yamlFilePath);
        return ParseYaml(content);
    }

    /// <summary>
    /// Parse YAML content string (simple parser for inference.yml format)
    /// </summary>
    public static Dictionary<string, object> ParseYaml(string yamlContent)
    {
        var result = new Dictionary<string, object>(StringComparer.OrdinalIgnoreCase);
        var lines = yamlContent.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);

        string? currentKey = null;
        var currentList = new List<object>();
        var isInList = false;

        foreach (var line in lines)
        {
            var trimmed = line.Trim();
            
            // Skip comments and empty lines
            if (trimmed.StartsWith("#") || string.IsNullOrWhiteSpace(trimmed))
                continue;

            // List item
            if (trimmed.StartsWith("-"))
            {
                var value = trimmed.Substring(1).Trim();
                if (!string.IsNullOrEmpty(value))
                {
                    isInList = true;
                    currentList.Add(ParseValue(value));
                }
                continue;
            }

            // Key-value pair
            if (trimmed.Contains(":"))
            {
                // Save previous list if any
                if (isInList && currentKey != null)
                {
                    result[currentKey] = currentList.ToArray();
                    currentList = new List<object>();
                    isInList = false;
                }

                var parts = trimmed.Split(new[] { ':' }, 2);
                currentKey = parts[0].Trim();
                var value = parts.Length > 1 ? parts[1].Trim() : string.Empty;

                if (!string.IsNullOrEmpty(value))
                {
                    result[currentKey] = ParseValue(value);
                    currentKey = null;
                }
                // else value might be on next lines (list or nested object)
            }
        }

        // Save last list if any
        if (isInList && currentKey != null)
        {
            result[currentKey] = currentList.ToArray();
        }

        return result;
    }

    private static object ParseValue(string value)
    {
        // Remove quotes
        value = value.Trim().Trim('"', '\'');

        // Try parse as number
        if (int.TryParse(value, out var intVal))
            return intVal;
        if (double.TryParse(value, out var doubleVal))
            return doubleVal;

        // Boolean
        if (value.Equals("true", StringComparison.OrdinalIgnoreCase))
            return true;
        if (value.Equals("false", StringComparison.OrdinalIgnoreCase))
            return false;

        // List in brackets
        if (value.StartsWith("[") && value.EndsWith("]"))
        {
            var items = value.Substring(1, value.Length - 2)
                            .Split(',')
                            .Select(s => ParseValue(s.Trim()))
                            .ToArray();
            return items;
        }

        // String
        return value;
    }

    /// <summary>
    /// Get value from parsed YAML dictionary with default
    /// </summary>
    public static T GetValue<T>(Dictionary<string, object> config, string key, T defaultValue)
    {
        if (!config.TryGetValue(key, out var value))
            return defaultValue;

        try
        {
            if (value is T typedValue)
                return typedValue;

            return (T)Convert.ChangeType(value, typeof(T));
        }
        catch
        {
            return defaultValue;
        }
    }

    /// <summary>
    /// Get array value from parsed YAML dictionary
    /// </summary>
    public static T[] GetArray<T>(Dictionary<string, object> config, string key, T[]? defaultValue = null)
    {
        if (!config.TryGetValue(key, out var value))
            return defaultValue ?? Array.Empty<T>();

        if (value is T[] typedArray)
            return typedArray;

        if (value is object[] objArray)
        {
            try
            {
                return objArray.Select(o => (T)Convert.ChangeType(o, typeof(T))).ToArray();
            }
            catch
            {
                return defaultValue ?? Array.Empty<T>();
            }
        }

        return defaultValue ?? Array.Empty<T>();
    }
}
