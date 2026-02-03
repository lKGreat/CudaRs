using System;
using System.Runtime.InteropServices;

namespace CudaRS.Core;

public static class ThreadAffinity
{
    public static bool TrySetCurrentThreadAffinity(int[] cores)
    {
        if (cores == null || cores.Length == 0)
            return false;

        if (!OperatingSystem.IsWindows())
            return false;

        ulong mask = 0;
        foreach (var core in cores)
        {
            if (core < 0 || core >= 64)
                continue;
            mask |= 1UL << core;
        }

        if (mask == 0)
            return false;

        var handle = GetCurrentThread();
        var prev = SetThreadAffinityMask(handle, new UIntPtr(mask));
        return prev != UIntPtr.Zero;
    }

    [DllImport("kernel32.dll")]
    private static extern IntPtr GetCurrentThread();

    [DllImport("kernel32.dll")]
    private static extern UIntPtr SetThreadAffinityMask(IntPtr hThread, UIntPtr dwThreadAffinityMask);
}
